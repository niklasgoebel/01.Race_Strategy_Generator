# src/outputs/export_mobile.py
"""
Mobile-optimized HTML and JSON export functionality.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def generate_mobile_html(result: Any, profile: Dict[str, Any]) -> str:
    """
    Generate a mobile-optimized HTML page for race strategy.
    Single column, large fonts, high contrast, simplified tables.
    
    Args:
        result: PipelineResult with course analysis and strategy
        profile: Athlete profile dictionary
    
    Returns:
        HTML string
    """
    athlete_name = profile.get("name", "Athlete")
    race_name = result.course_summary.get("race_name", "Race")
    distance = result.course_summary.get("total_distance_km", "-")
    gain = result.course_summary.get("total_gain_m", "-")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{race_name} Strategy - {athlete_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            font-size: 18px;
            line-height: 1.6;
            background: #000;
            color: #fff;
            padding: 20px;
            max-width: 100%;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 20px;
            margin: -20px -20px 30px -20px;
            text-align: center;
        }}
        
        h1 {{
            font-size: 28px;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .subtitle {{
            font-size: 16px;
            opacity: 0.9;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            background: #1a1a1a;
            padding: 20px;
            margin: 20px 0;
            border-radius: 12px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: 700;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 14px;
            opacity: 0.7;
            margin-top: 5px;
        }}
        
        .section {{
            margin: 30px 0;
        }}
        
        h2 {{
            font-size: 24px;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .strategy-text {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            font-size: 16px;
            line-height: 1.8;
        }}
        
        .critical-section {{
            background: #2a1a3a;
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #fff;
        }}
        
        .section-detail {{
            font-size: 16px;
            margin: 8px 0;
            opacity: 0.9;
        }}
        
        .effort {{
            display: inline-block;
            background: #667eea;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .fueling {{
            background: #1a3a1a;
            border-left: 4px solid #4ade80;
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
        }}
        
        .mental-cue {{
            background: #3a2a1a;
            border-left: 4px solid #fbbf24;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }}
        
        .km-marker {{
            display: inline-block;
            background: #fbbf24;
            color: #000;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 700;
            margin-right: 10px;
        }}
        
        ul {{
            list-style: none;
            padding: 0;
        }}
        
        li {{
            padding: 10px 0;
            padding-left: 25px;
            position: relative;
        }}
        
        li:before {{
            content: "•";
            position: absolute;
            left: 10px;
            color: #667eea;
            font-size: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{race_name}</h1>
        <div class="subtitle">{athlete_name}</div>
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{distance}</div>
            <div class="stat-label">km</div>
        </div>
        <div class="stat">
            <div class="stat-value">{gain}</div>
            <div class="stat-label">gain (m)</div>
        </div>
    </div>
"""
    
    # Strategy text
    if result.strategy_text:
        html += """
    <div class="section">
        <h2>🎯 Strategy</h2>
        <div class="strategy-text">
"""
        # Parse strategy text into bullets
        lines = result.strategy_text.split("\n")
        html += "<ul>"
        for line in lines:
            line = line.strip()
            if line and (line.startswith("- ") or line.startswith("• ")):
                text = line.lstrip("-").lstrip("•").strip()
                html += f"<li>{text}</li>"
        html += "</ul>"
        html += """
        </div>
    </div>
"""
    
    # Critical Sections
    if result.strategy_data and "critical_sections" in result.strategy_data:
        sections = result.strategy_data["critical_sections"]
        if sections:
            html += """
    <div class="section">
        <h2>⚠️ Critical Sections</h2>
"""
            for section in sections:
                label = section.get("label", "Section")
                start_km = section.get("start_km", 0)
                end_km = section.get("end_km", 0)
                gain_m = section.get("gain_m", 0)
                gradient = section.get("avg_gradient_pct", 0)
                effort_rpe = section.get("effort_rpe", "-")
                notes = section.get("notes", "")
                
                html += f"""
        <div class="critical-section">
            <div class="section-title">{label}</div>
            <div class="section-detail">📍 km {start_km:.1f} - {end_km:.1f}</div>
            <div class="section-detail">⛰️ +{gain_m:.0f}m ({gradient:.1f}% grade)</div>
            <div class="effort">RPE {effort_rpe}</div>
            {f'<div class="section-detail" style="margin-top: 15px;">{notes}</div>' if notes else ''}
        </div>
"""
            html += """
    </div>
"""
    
    # Fueling
    if result.strategy_data and "fueling_plan" in result.strategy_data:
        fueling = result.strategy_data["fueling_plan"]
        carbs = fueling.get("carbs_g_per_hour", "-")
        hydration = fueling.get("hydration_notes", "")
        
        html += f"""
    <div class="section">
        <h2>🍌 Fueling</h2>
        <div class="fueling">
            <div class="section-title">{carbs} g carbs/hour</div>
            {f'<div class="section-detail">{hydration}</div>' if hydration else ''}
        </div>
    </div>
"""
    
    # Mental Cues
    if result.strategy_data and "mental_cues" in result.strategy_data:
        cues = result.strategy_data["mental_cues"]
        if cues:
            html += """
    <div class="section">
        <h2>💭 Mental Cues</h2>
"""
            for cue in cues:
                km = cue.get("km", 0)
                text = cue.get("cue", "")
                html += f"""
        <div class="mental-cue">
            <span class="km-marker">km {km:.0f}</span>
            <span>{text}</span>
        </div>
"""
            html += """
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    return html


def generate_json_export(result: Any, profile: Dict[str, Any]) -> str:
    """
    Generate a JSON export of the complete strategy data.
    
    Args:
        result: PipelineResult with course analysis and strategy
        profile: Athlete profile dictionary
    
    Returns:
        JSON string
    """
    export_data = {
        "athlete_profile": profile,
        "course_summary": result.course_summary,
        "strategy_text": result.strategy_text,
        "strategy_data": result.strategy_data,
        "segments": result.segments_df.to_dict(orient="records") if hasattr(result, "segments_df") else [],
        "key_climbs": result.climbs_df.to_dict(orient="records") if hasattr(result, "climbs_df") else [],
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

