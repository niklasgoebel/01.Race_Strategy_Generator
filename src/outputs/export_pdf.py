# src/outputs/export_pdf.py
"""
PDF export functionality for race strategies.
Generates full strategy PDFs and race-day cheat sheets.
"""

from __future__ import annotations

import io
from typing import Any, Dict

from fpdf import FPDF


class RaceStrategyPDF(FPDF):
    """Custom PDF class for race strategy documents."""

    def __init__(self, athlete_name: str, race_name: str):
        super().__init__()
        self.athlete_name = athlete_name
        self.race_name = race_name

    def header(self):
        """PDF header with title."""
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, f"Race Strategy: {self.race_name}", 0, 1, "C")
        self.set_font("Arial", "I", 10)
        self.cell(0, 5, f"Athlete: {self.athlete_name}", 0, 1, "C")
        self.ln(5)

    def footer(self):
        """PDF footer with page number."""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def _sanitize_text(text: str) -> str:
    """Remove Unicode characters that aren't supported by basic PDF fonts."""
    # Remove emojis and other special Unicode characters
    # Keep only ASCII-compatible characters
    return ''.join(char for char in text if ord(char) < 128 or char in ['é', 'è', 'à', 'ü', 'ö', 'ä'])


def generate_race_day_pdf(result: Any, profile: Dict[str, Any]) -> bytes:
    """
    Generate a full race strategy PDF.
    
    Args:
        result: PipelineResult with course analysis and strategy
        profile: Athlete profile dictionary
    
    Returns:
        PDF file as bytes
    """
    athlete_name = _sanitize_text(profile.get("name", "Athlete"))
    race_name = _sanitize_text(result.course_summary.get("race_name", "Race"))
    
    pdf = RaceStrategyPDF(athlete_name, race_name)
    pdf.add_page()
    
    # Course Overview
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Course Overview", 0, 1)
    pdf.set_font("Arial", "", 11)
    
    distance = result.course_summary.get("total_distance_km", "-")
    gain = result.course_summary.get("total_gain_m", "-")
    loss = result.course_summary.get("total_loss_m", "-")
    
    pdf.cell(0, 8, f"Distance: {distance} km", 0, 1)
    pdf.cell(0, 8, f"Elevation Gain: {gain} m", 0, 1)
    pdf.cell(0, 8, f"Elevation Loss: {loss} m", 0, 1)
    pdf.ln(5)
    
    # Strategy Text
    if result.strategy_text:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Race Strategy", 0, 1)
        pdf.set_font("Arial", "", 10)
        
        # Handle multi-line text
        strategy_lines = result.strategy_text.split("\n")
        for line in strategy_lines:
            if line.strip():
                # Sanitize line text
                line_clean = _sanitize_text(line)
                # Handle bullet points and formatting
                if line_clean.strip().startswith("- ") or line_clean.strip().startswith("• "):
                    pdf.set_x(15)
                    pdf.multi_cell(0, 5, line_clean.strip())
                elif line_clean.strip().startswith("#"):
                    # Section header
                    pdf.ln(3)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 6, line_clean.strip().replace("#", "").strip(), 0, 1)
                    pdf.set_font("Arial", "", 10)
                else:
                    pdf.multi_cell(0, 5, line_clean)
        
        pdf.ln(5)
    
    # Critical Sections
    if result.strategy_data and "critical_sections" in result.strategy_data:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Critical Sections", 0, 1)
        pdf.set_font("Arial", "", 9)
        
        for section in result.strategy_data["critical_sections"]:
            pdf.set_font("Arial", "B", 10)
            label = _sanitize_text(section.get("label", "Section"))
            pdf.cell(0, 7, label, 0, 1)
            
            pdf.set_font("Arial", "", 9)
            start_km = section.get("start_km", 0)
            end_km = section.get("end_km", 0)
            gain_m = section.get("gain_m", 0)
            gradient = section.get("avg_gradient_pct", 0)
            effort_rpe = section.get("effort_rpe", "-")
            effort_hr = section.get("effort_hr_percent_max", "-")
            notes = _sanitize_text(section.get("notes", ""))
            
            pdf.cell(0, 5, f"  {start_km:.1f} - {end_km:.1f} km | +{gain_m:.0f}m | {gradient:.1f}% grade", 0, 1)
            pdf.cell(0, 5, f"  Effort: RPE {effort_rpe} | HR {effort_hr}", 0, 1)
            if notes:
                pdf.set_x(10)
                pdf.multi_cell(0, 5, f"  {notes}")
            pdf.ln(2)
    
    # Fueling Plan
    if result.strategy_data and "fueling_plan" in result.strategy_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Fueling Plan", 0, 1)
        pdf.set_font("Arial", "", 10)
        
        fueling = result.strategy_data["fueling_plan"]
        carbs = fueling.get("carbs_g_per_hour", "-")
        hydration = _sanitize_text(fueling.get("hydration_notes", ""))
        
        pdf.cell(0, 7, f"Target: {carbs} g carbs/hour", 0, 1)
        if hydration:
            pdf.multi_cell(0, 5, f"Hydration: {hydration}")
        pdf.ln(3)
        
        if "special_sections" in fueling and fueling["special_sections"]:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, "Special Fueling Sections:", 0, 1)
            pdf.set_font("Arial", "", 9)
            
            for spec in fueling["special_sections"]:
                km_range = _sanitize_text(spec.get("km_range", ""))
                reason = _sanitize_text(spec.get("reason", ""))
                focus = _sanitize_text(spec.get("fueling_focus", ""))
                
                pdf.cell(0, 5, f"  km {km_range}: {reason}", 0, 1)
                pdf.set_x(15)
                pdf.multi_cell(0, 5, f"    {focus}")
    
    # Mental Cues
    if result.strategy_data and "mental_cues" in result.strategy_data:
        cues = result.strategy_data["mental_cues"]
        if cues:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Mental Cues", 0, 1)
            pdf.set_font("Arial", "", 10)
            
            for cue in cues:
                km = cue.get("km", 0)
                text = _sanitize_text(cue.get("cue", ""))
                pdf.cell(0, 6, f"km {km:.1f}: {text}", 0, 1)
    
    # Generate PDF bytes
    return pdf.output(dest="S").encode("latin1")


def generate_cheat_sheet_pdf(result: Any, profile: Dict[str, Any]) -> bytes:
    """
    Generate a condensed 1-page race day cheat sheet.
    
    Args:
        result: PipelineResult with course analysis and strategy
        profile: Athlete profile dictionary
    
    Returns:
        PDF file as bytes
    """
    athlete_name = _sanitize_text(profile.get("name", "Athlete"))
    race_name = _sanitize_text(result.course_summary.get("race_name", "Race"))
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 8, f"{race_name} - Cheat Sheet", 0, 1, "C")
    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 5, athlete_name, 0, 1, "C")
    pdf.ln(3)
    
    # Course Stats (compact)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Course", 0, 1)
    pdf.set_font("Arial", "", 9)
    
    distance = result.course_summary.get("total_distance_km", "-")
    gain = result.course_summary.get("total_gain_m", "-")
    
    pdf.cell(0, 5, f"{distance} km | +{gain}m", 0, 1)
    pdf.ln(2)
    
    # Key Strategy Points (bullets)
    if result.strategy_text:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Key Points", 0, 1)
        pdf.set_font("Arial", "", 8)
        
        # Extract bullet points from strategy text
        lines = result.strategy_text.split("\n")
        bullet_count = 0
        for line in lines:
            if (line.strip().startswith("- ") or line.strip().startswith("• ")) and bullet_count < 8:
                text = _sanitize_text(line.strip().lstrip("-").lstrip("•").strip())
                pdf.set_x(12)
                pdf.multi_cell(0, 4, f"• {text}")
                bullet_count += 1
        
        pdf.ln(2)
    
    # Critical Sections (compact table)
    if result.strategy_data and "critical_sections" in result.strategy_data:
        sections = result.strategy_data["critical_sections"]
        if sections:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, "Critical Sections", 0, 1)
            pdf.set_font("Arial", "", 7)
            
            for i, section in enumerate(sections[:5]):  # Max 5 sections
                label = _sanitize_text(section.get("label", f"Section {i+1}"))
                start_km = section.get("start_km", 0)
                end_km = section.get("end_km", 0)
                effort = section.get("effort_rpe", "-")
                
                pdf.cell(0, 4, f"{label}: km {start_km:.1f}-{end_km:.1f} | RPE {effort}", 0, 1)
            
            pdf.ln(2)
    
    # Fueling (compact)
    if result.strategy_data and "fueling_plan" in result.strategy_data:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Fueling", 0, 1)
        pdf.set_font("Arial", "", 8)
        
        fueling = result.strategy_data["fueling_plan"]
        carbs = fueling.get("carbs_g_per_hour", "-")
        pdf.cell(0, 4, f"{carbs} g carbs/hour", 0, 1)
    
    # Mental Cues (compact)
    if result.strategy_data and "mental_cues" in result.strategy_data:
        cues = result.strategy_data["mental_cues"]
        if cues:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, "Mental Cues", 0, 1)
            pdf.set_font("Arial", "", 7)
            
            for cue in cues[:6]:  # Max 6 cues
                km = cue.get("km", 0)
                text = _sanitize_text(cue.get("cue", ""))
                pdf.cell(0, 4, f"km {km:.0f}: {text}", 0, 1)
    
    return pdf.output(dest="S").encode("latin1")

