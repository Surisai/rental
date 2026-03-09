"""
Logging utilities for the rental data cleaning pipeline.
"""

from datetime import datetime


def initialize_logging():
    """Initialize logging system"""
    log_content = []
    log_content.append("="*70)
    log_content.append("CANADIAN RENTAL DATA CLEANING LOG")
    log_content.append(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append("="*70 + "\n")
    return log_content


def log_step(log_content, step_num, step_name, details=""):
    """Add a step to the log"""
    log_content.append(f"\nSTEP {step_num}: {step_name}")
    log_content.append("-"*40)
    if details:
        if isinstance(details, list):
            log_content.extend(details)
        else:
            log_content.append(details)
    return log_content


def save_log(log_content, log_path):
    """Save log to file"""
    with open(log_path, 'w') as f:
        f.write("\n".join(log_content))
    print(f"✓ Cleaning log saved to: {log_path}")