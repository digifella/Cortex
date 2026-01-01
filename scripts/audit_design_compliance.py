#!/usr/bin/env python3
"""
Design System Compliance Auditor
Checks all pages against DESIGN_SYSTEM_GUIDE.md standards
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import json


class DesignAuditor:
    """Audits pages for design system compliance"""

    def __init__(self, pages_dir: Path):
        self.pages_dir = pages_dir
        self.results = {}

    def audit_page(self, page_path: Path) -> Dict:
        """Audit a single page against design standards"""
        content = page_path.read_text()

        checks = {
            "has_apply_theme": bool(re.search(r'apply_theme\(\)', content)),
            "has_page_config": bool(re.search(r'st\.set_page_config', content)),
            "has_page_icon": bool(re.search(r'page_icon\s*=', content)),
            "has_title": bool(re.search(r'st\.title\(', content)),
            "has_nav_caption": bool(re.search(r'sidebar.*navigate', content, re.DOTALL | re.IGNORECASE)),
            "has_divider_after_header": bool(re.search(r'st\.markdown\("---"\)', content)),
            "uses_section_header": bool(re.search(r'section_header\(', content)),
            "uses_error_display": bool(re.search(r'error_display\(', content)),
            "has_version_footer": bool(re.search(r'render_version_footer\(\)', content)),
            "has_logger": bool(re.search(r'get_logger\(__name__\)', content)),
            "has_help_text": bool(re.search(r'help\s*=', content)),
            "has_wide_layout": bool(re.search(r'layout\s*=\s*["\']wide["\']', content)),
        }

        # Calculate compliance score
        score = (sum(checks.values()) / len(checks)) * 100

        # Extract version info
        version_match = re.search(r'Version:\s*v?([\d.]+)', content)
        date_match = re.search(r'Date:\s*([\d-]+)', content)

        return {
            "file": page_path.name,
            "score": round(score, 1),
            "checks": checks,
            "version": version_match.group(1) if version_match else "Unknown",
            "date": date_match.group(1) if date_match else "Unknown",
            "lines": len(content.splitlines())
        }

    def audit_all_pages(self) -> Dict:
        """Audit all pages in the pages directory"""
        pages = sorted(self.pages_dir.glob("*.py"))

        for page in pages:
            if page.name.startswith("_"):  # Skip private/component files
                continue
            self.results[page.name] = self.audit_page(page)

        return self.results

    def generate_report(self) -> str:
        """Generate a comprehensive audit report"""
        if not self.results:
            self.audit_all_pages()

        # Sort by score (lowest first - needs most work)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['score']
        )

        report = []
        report.append("=" * 80)
        report.append("🎨 CORTEX SUITE DESIGN SYSTEM COMPLIANCE AUDIT")
        report.append("=" * 80)
        report.append("")

        # Overall stats
        total_pages = len(sorted_results)
        avg_score = sum(r['score'] for _, r in sorted_results) / total_pages if total_pages > 0 else 0
        fully_compliant = sum(1 for _, r in sorted_results if r['score'] == 100)

        report.append(f"📊 OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Pages: {total_pages}")
        report.append(f"Average Compliance: {avg_score:.1f}%")
        report.append(f"Fully Compliant: {fully_compliant} ({(fully_compliant/total_pages*100):.1f}%)")
        report.append(f"Need Attention: {total_pages - fully_compliant}")
        report.append("")

        # Pages needing most attention (score < 80%)
        critical = [(name, r) for name, r in sorted_results if r['score'] < 80]
        if critical:
            report.append(f"🚨 CRITICAL - Need Immediate Attention ({len(critical)} pages)")
            report.append("-" * 80)
            for name, result in critical:
                report.append(f"  {result['score']:5.1f}% - {name:50s} v{result['version']}")
            report.append("")

        # Moderate issues (80-95%)
        moderate = [(name, r) for name, r in sorted_results if 80 <= r['score'] < 95]
        if moderate:
            report.append(f"⚠️  MODERATE - Minor Updates Needed ({len(moderate)} pages)")
            report.append("-" * 80)
            for name, result in moderate:
                report.append(f"  {result['score']:5.1f}% - {name:50s} v{result['version']}")
            report.append("")

        # Good compliance (95-100%)
        good = [(name, r) for name, r in sorted_results if r['score'] >= 95]
        if good:
            report.append(f"✅ GOOD - Near Perfect ({len(good)} pages)")
            report.append("-" * 80)
            for name, result in good:
                report.append(f"  {result['score']:5.1f}% - {name:50s} v{result['version']}")
            report.append("")

        # Detailed page-by-page breakdown
        report.append("=" * 80)
        report.append("📋 DETAILED PAGE-BY-PAGE ANALYSIS")
        report.append("=" * 80)
        report.append("")

        for name, result in sorted_results:
            report.append(f"📄 {name}")
            report.append(f"   Score: {result['score']:.1f}%  |  Version: v{result['version']}  |  Date: {result['date']}  |  Lines: {result['lines']}")
            report.append("")

            # Show what's missing
            missing = [check for check, passed in result['checks'].items() if not passed]
            if missing:
                report.append("   ❌ Missing:")
                for item in missing:
                    report.append(f"      • {item}")
                report.append("")
            else:
                report.append("   ✅ All checks passed!")
                report.append("")

            report.append("-" * 80)

        # Recommendations
        report.append("")
        report.append("=" * 80)
        report.append("💡 RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")

        if critical:
            report.append("1. **Priority 1**: Update critical pages (score < 80%)")
            for name, _ in critical[:3]:  # Show top 3
                report.append(f"   • {name}")
            report.append("")

        if moderate:
            report.append("2. **Priority 2**: Polish moderate pages (score 80-95%)")
            report.append(f"   • {len(moderate)} pages need minor updates")
            report.append("")

        # Common issues
        all_checks = {}
        for result in self.results.values():
            for check, passed in result['checks'].items():
                if check not in all_checks:
                    all_checks[check] = {'passed': 0, 'failed': 0}
                if passed:
                    all_checks[check]['passed'] += 1
                else:
                    all_checks[check]['failed'] += 1

        report.append("3. **Common Issues** (most frequent failures):")
        failed_checks = sorted(
            [(check, stats['failed']) for check, stats in all_checks.items()],
            key=lambda x: x[1],
            reverse=True
        )
        for check, count in failed_checks[:5]:
            if count > 0:
                report.append(f"   • {check}: {count} pages missing ({(count/total_pages*100):.1f}%)")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, output_path: Path):
        """Save audit report to file"""
        report = self.generate_report()
        output_path.write_text(report)
        print(f"✅ Report saved to: {output_path}")
        return report


if __name__ == "__main__":
    # Run audit
    pages_dir = Path(__file__).parent.parent / "pages"
    auditor = DesignAuditor(pages_dir)

    # Generate and save report
    output_path = Path(__file__).parent.parent / "docs" / "DESIGN_COMPLIANCE_AUDIT.md"
    report = auditor.save_report(output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("🎨 Design System Compliance Audit Complete!")
    print("=" * 80)

    # Print to console too
    print("\n" + report)

    # Save JSON for programmatic access
    json_path = Path(__file__).parent.parent / "docs" / "design_audit.json"
    json_path.write_text(json.dumps(auditor.results, indent=2))
    print(f"\n✅ JSON data saved to: {json_path}")
