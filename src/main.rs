//! Extract ECG waveform data from KardiaMobile 6-lead ECG PDF and save as EDF.
//!
//! Reads the vector path data embedded in the PDF, converts path coordinates
//! to voltage values using the known calibration (10mm/mV, 25mm/s), and writes
//! the result as a standard EDF file.

use anyhow::{bail, Context, Result};
use lopdf::content::{Content, Operation};
use lopdf::Document;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Calibration: 1 mV = 28.346 PDF points (10mm at 2.8346 pt/mm).
const CAL_PT_PER_MV: f64 = 28.346;

/// KardiaMobile 6L samples at 300 Hz.
const SAMPLE_RATE: usize = 300;

/// Number of ECG leads.
const NUM_LEADS: usize = 6;

/// Lead names in order.
const LEAD_NAMES: [&str; NUM_LEADS] = ["I", "II", "III", "aVR", "aVL", "aVF"];

// ---------------------------------------------------------------------------
// PDF content stream parsing
// ---------------------------------------------------------------------------

/// A single drawn path with its associated graphics state.
struct DrawnPath {
    line_width: f64,
    points: Vec<(f64, f64)>,
    is_stroked: bool,
}

/// Extract an f64 from a lopdf Object.
fn obj_to_f64(obj: &lopdf::Object) -> Option<f64> {
    match obj {
        lopdf::Object::Integer(i) => Some(*i as f64),
        lopdf::Object::Real(f) => Some(*f as f64),
        _ => None,
    }
}

/// Parse one page's content stream and return all stroked paths.
///
/// The KardiaMobile PDF wraps drawing ops in `q 1 0 0 -1 0 792 cm ... S Q`,
/// which means the coordinates in the content stream are in screen-coordinate
/// space (origin top-left, y increasing downward). We use them directly.
fn extract_paths_from_ops(ops: &[Operation]) -> Vec<DrawnPath> {
    let mut paths = Vec::new();
    let mut line_width: f64 = 1.0;
    let mut current_points: Vec<(f64, f64)> = Vec::new();
    let mut width_stack: Vec<f64> = Vec::new();

    for op in ops {
        match op.operator.as_str() {
            // Graphics state
            "q" => width_stack.push(line_width),
            "Q" => {
                if let Some(w) = width_stack.pop() {
                    line_width = w;
                }
            }
            "w" => {
                if let Some(w) = op.operands.first().and_then(obj_to_f64) {
                    line_width = w;
                }
            }
            // We ignore "cm" because the coordinates inside the cm block
            // are already in screen space for this PDF.

            // Path construction
            "m" => {
                // moveto — start a new sub-path
                if let (Some(x), Some(y)) = (
                    op.operands.first().and_then(obj_to_f64),
                    op.operands.get(1).and_then(obj_to_f64),
                ) {
                    current_points.push((x, y));
                }
            }
            "l" => {
                // lineto
                if let (Some(x), Some(y)) = (
                    op.operands.first().and_then(obj_to_f64),
                    op.operands.get(1).and_then(obj_to_f64),
                ) {
                    current_points.push((x, y));
                }
            }
            // Painting
            "S" | "s" => {
                // stroke (or close-and-stroke)
                if !current_points.is_empty() {
                    paths.push(DrawnPath {
                        line_width,
                        points: std::mem::take(&mut current_points),
                        is_stroked: true,
                    });
                }
            }
            "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" => {
                // fill variants — not ECG waveforms, discard
                current_points.clear();
            }
            "n" => {
                // end path without painting
                current_points.clear();
            }
            _ => {}
        }
    }

    paths
}

// ---------------------------------------------------------------------------
// ECG extraction logic
// ---------------------------------------------------------------------------

/// Find the 6 baseline y-coordinates from the horizontal grid lines.
///
/// Baselines are long horizontal lines (x-span > 500) drawn at line_width ≈ 0.4.
fn find_baselines(paths: &[DrawnPath]) -> Option<[f64; NUM_LEADS]> {
    for path in paths {
        if !(0.35..0.45).contains(&path.line_width) {
            continue;
        }
        // Collect y-values of horizontal lines that span the page
        let mut y_values: Vec<f64> = Vec::new();
        for pair in path.points.windows(2) {
            let (x1, y1) = pair[0];
            let (x2, y2) = pair[1];
            if (y1 - y2).abs() < 0.01 && (x2 - x1).abs() > 500.0 {
                y_values.push(y1);
            }
        }
        if y_values.len() >= NUM_LEADS {
            let mut baselines = [0.0; NUM_LEADS];
            baselines.copy_from_slice(&y_values[..NUM_LEADS]);
            return Some(baselines);
        }
    }
    None
}

/// Classify a path as an ECG waveform segment and return its lead index.
///
/// ECG segments have line_width ≈ 0.4 and many (>50) line-to points.
fn classify_ecg_path(path: &DrawnPath, baselines: &[f64; NUM_LEADS]) -> Option<usize> {
    if !(0.35..0.45).contains(&path.line_width) || path.points.len() < 50 {
        return None;
    }

    // Compute y-center of all points
    let y_sum: f64 = path.points.iter().map(|p| p.1).sum();
    let y_center = y_sum / path.points.len() as f64;

    // Find nearest baseline
    let mut best_lead = 0;
    let mut min_dist = f64::MAX;
    for (i, &bl) in baselines.iter().enumerate() {
        let dist = (y_center - bl).abs();
        if dist < min_dist {
            min_dist = dist;
            best_lead = i;
        }
    }

    if min_dist < 50.0 {
        Some(best_lead)
    } else {
        None
    }
}

/// Convert y-coordinate points to millivolt values.
///
/// In screen coordinates, y increases downward, so:
///   voltage_mV = (baseline_y - y) / CAL_PT_PER_MV
fn points_to_mv(points: &[(f64, f64)], baseline_y: f64) -> Vec<f64> {
    points
        .iter()
        .map(|&(_, y)| (baseline_y - y) / CAL_PT_PER_MV)
        .collect()
}

// ---------------------------------------------------------------------------
// EDF file writer
// ---------------------------------------------------------------------------

/// Write a fixed-width ASCII field, right-padded with spaces.
fn write_field(buf: &mut Vec<u8>, value: &str, width: usize) {
    let bytes = value.as_bytes();
    let len = bytes.len().min(width);
    buf.extend_from_slice(&bytes[..len]);
    for _ in len..width {
        buf.push(b' ');
    }
}

/// Write ECG data as an EDF file.
fn write_edf(path: &Path, lead_data: &[Vec<f64>; NUM_LEADS]) -> Result<()> {
    let n_signals = NUM_LEADS;
    let record_duration_secs = 1;
    let samples_per_record = SAMPLE_RATE * record_duration_secs;

    // Pad all channels to a whole number of records
    let max_samples = lead_data.iter().map(|v| v.len()).max().unwrap_or(0);
    let n_records = (max_samples + samples_per_record - 1) / samples_per_record;

    // Compute physical min/max per channel (with 0.1 mV margin)
    let mut phys_min = [0.0f64; NUM_LEADS];
    let mut phys_max = [0.0f64; NUM_LEADS];
    for (i, data) in lead_data.iter().enumerate() {
        let lo = data.iter().cloned().fold(f64::MAX, f64::min);
        let hi = data.iter().cloned().fold(f64::MIN, f64::max);
        phys_min[i] = lo - 0.1;
        phys_max[i] = hi + 0.1;
    }

    let dig_min: i16 = -32768;
    let dig_max: i16 = 32767;

    // Build header
    let header_bytes = 256 + n_signals * 256;
    let mut hdr = Vec::with_capacity(header_bytes);

    // -- General header (256 bytes) --
    write_field(&mut hdr, "0", 8); // version
    // Patient ID: code sex dob name (EDF+ spec)
    write_field(&mut hdr, "X M 04-MAY-1970 Joel_Henderson", 80);
    // Recording ID
    write_field(&mut hdr, "Startdate 23-FEB-2026 X KardiaMobile_6L", 80);
    write_field(&mut hdr, "23.02.26", 8); // start date dd.mm.yy
    write_field(&mut hdr, "18.26.56", 8); // start time hh.mm.ss
    write_field(&mut hdr, &header_bytes.to_string(), 8);
    write_field(&mut hdr, "", 44); // reserved (plain EDF)
    write_field(&mut hdr, &n_records.to_string(), 8);
    write_field(&mut hdr, &record_duration_secs.to_string(), 8);
    write_field(&mut hdr, &n_signals.to_string(), 4);

    // -- Per-signal headers (n_signals × 256 bytes total) --
    // Labels (16 each)
    for name in &LEAD_NAMES {
        write_field(&mut hdr, &format!("EKG {}", name), 16);
    }
    // Transducer type (80 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, "KardiaMobile 6L electrode", 80);
    }
    // Physical dimension (8 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, "mV", 8);
    }
    // Physical minimum (8 each)
    for i in 0..n_signals {
        write_field(&mut hdr, &format!("{:.4}", phys_min[i]), 8);
    }
    // Physical maximum (8 each)
    for i in 0..n_signals {
        write_field(&mut hdr, &format!("{:.4}", phys_max[i]), 8);
    }
    // Digital minimum (8 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, &dig_min.to_string(), 8);
    }
    // Digital maximum (8 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, &dig_max.to_string(), 8);
    }
    // Prefiltering (80 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, "Enhanced Filter, 50Hz mains", 80);
    }
    // Samples per data record (8 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, &samples_per_record.to_string(), 8);
    }
    // Reserved (32 each)
    for _ in 0..n_signals {
        write_field(&mut hdr, "", 32);
    }

    assert_eq!(hdr.len(), header_bytes, "Header size mismatch");

    // Build data records
    // Each record: for each signal, samples_per_record × 2 bytes (i16 LE)
    let record_bytes = n_signals * samples_per_record * 2;
    let mut data_buf = Vec::with_capacity(n_records * record_bytes);

    for rec in 0..n_records {
        let offset = rec * samples_per_record;
        for ch in 0..n_signals {
            let channel = &lead_data[ch];
            let scale =
                (dig_max as f64 - dig_min as f64) / (phys_max[ch] - phys_min[ch]);
            for s in 0..samples_per_record {
                let idx = offset + s;
                let phys = if idx < channel.len() {
                    channel[idx]
                } else {
                    // Pad with last sample value
                    channel.last().copied().unwrap_or(0.0)
                };
                let digital = ((phys - phys_min[ch]) * scale + dig_min as f64)
                    .round()
                    .clamp(dig_min as f64, dig_max as f64)
                    as i16;
                data_buf.extend_from_slice(&digital.to_le_bytes());
            }
        }
    }

    // Write file
    let mut file = File::create(path)?;
    file.write_all(&hdr)?;
    file.write_all(&data_buf)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let base_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let pdf_path = base_dir.join("kardiamobile-6-lead-ecg.pdf");
    let edf_path = base_dir.join("kardiamobile-6-lead-ecg.edf");

    let doc = Document::load(&pdf_path)
        .with_context(|| format!("Failed to open PDF: {}", pdf_path.display()))?;

    // ECG data is on pages 2–5 (page numbers, 1-indexed in PDF)
    let pages = doc.get_pages();
    let mut page_ids: Vec<_> = pages.iter().collect();
    page_ids.sort_by_key(|(num, _)| *num);

    // Extract baselines from page 2 (index 1 in sorted list)
    let (_, &page2_id) = page_ids[1];
    let content_data = doc
        .get_page_content(page2_id)
        .context("Failed to read page 2 content")?;
    let content = Content::decode(&content_data)
        .context("Failed to decode page 2 content stream")?;
    let page2_paths = extract_paths_from_ops(&content.operations);
    let baselines =
        find_baselines(&page2_paths).context("Could not find baseline grid lines in PDF")?;

    println!("Baselines (PDF y-coordinates):");
    for (i, bl) in baselines.iter().enumerate() {
        println!("  Lead {:3}: {:.3}", LEAD_NAMES[i], bl);
    }

    // Extract waveforms from pages 2–5 (indices 1..=4)
    let mut all_leads: [Vec<(f64, f64)>; NUM_LEADS] = Default::default();

    for &pg_idx in &[1usize, 2, 3, 4] {
        let (_, &page_id) = page_ids[pg_idx];
        let content_data = doc
            .get_page_content(page_id)
            .with_context(|| format!("Failed to read page {} content", pg_idx + 1))?;
        let content = Content::decode(&content_data)
            .with_context(|| format!("Failed to decode page {} content", pg_idx + 1))?;
        let paths = extract_paths_from_ops(&content.operations);

        // Collect this page's paths per lead, sorted by x within the page
        let mut page_leads: [Vec<(f64, f64)>; NUM_LEADS] = Default::default();
        for path in &paths {
            if !path.is_stroked {
                continue;
            }
            if let Some(lead_idx) = classify_ecg_path(path, &baselines) {
                page_leads[lead_idx].extend_from_slice(&path.points);
            }
        }
        // Sort within each page (segments may arrive out of order)
        for lead in &mut page_leads {
            lead.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        // Append page data in page order (do NOT sort globally,
        // because each page reuses the same x-coordinate range)
        for i in 0..NUM_LEADS {
            all_leads[i].extend_from_slice(&page_leads[i]);
        }
    }

    // Convert to voltages, deduplicating boundary points
    let mut lead_voltages: [Vec<f64>; NUM_LEADS] = Default::default();

    for i in 0..NUM_LEADS {
        let points = &all_leads[i];
        if points.is_empty() {
            bail!("No data extracted for lead {}", LEAD_NAMES[i]);
        }

        // Deduplicate points with nearly-identical x-coordinates
        let mut deduped: Vec<(f64, f64)> = vec![points[0]];
        for &pt in &points[1..] {
            if (pt.0 - deduped.last().unwrap().0).abs() > 0.01 {
                deduped.push(pt);
            }
        }

        let voltages = points_to_mv(&deduped, baselines[i]);
        let min_v = voltages.iter().cloned().fold(f64::MAX, f64::min);
        let max_v = voltages.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "Lead {:3}: {} samples, range [{:.3}, {:.3}] mV",
            LEAD_NAMES[i],
            voltages.len(),
            min_v,
            max_v
        );
        lead_voltages[i] = voltages;
    }

    // Trim all leads to the shortest length
    let min_len = lead_voltages.iter().map(|v| v.len()).min().unwrap();
    for lead in &mut lead_voltages {
        lead.truncate(min_len);
    }

    let duration_sec = min_len as f64 / SAMPLE_RATE as f64;
    println!("\nTotal samples per lead: {}", min_len);
    println!("Duration: {:.2} seconds", duration_sec);
    println!("Sampling rate: {} Hz", SAMPLE_RATE);

    // Write EDF
    write_edf(&edf_path, &lead_voltages)?;

    let file_size = std::fs::metadata(&edf_path)?.len();
    println!("\nEDF file written: {}", edf_path.display());
    println!("File size: {} bytes", file_size);

    Ok(())
}
