import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime


class TEDDataValidator:
    """Validate TED dataset for RAG pipeline readiness"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.errors = []
        self.warnings = []
        self.stats = {}

    def load_data(self) -> bool:
        """Load and perform initial validation"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded {len(self.df)} talks")
            return True
        except Exception as e:
            self.errors.append(f"Failed to load CSV: {e}")
            return False

    def validate_required_columns(self) -> bool:
        """Check all required columns exist"""
        required = ['talk_id', 'title', 'speaker_1', 'transcript', 'description', 'topics']
        missing = [col for col in required if col not in self.df.columns]

        if missing:
            self.errors.append(f"Missing required columns: {missing}")
            return False

        print(f"✅ All required columns present")
        return True

    def validate_critical_fields(self):
        """Check critical fields for nulls and validity"""

        # Check for nulls in critical fields
        critical = ['talk_id', 'title', 'transcript']
        for col in critical:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                self.errors.append(f"{col}: {null_count} null values")

        # Check for empty transcripts
        empty_transcripts = (self.df['transcript'].str.strip() == '').sum()
        if empty_transcripts > 0:
            self.errors.append(f"{empty_transcripts} empty transcripts")

        # Validate talk_id uniqueness
        duplicate_ids = self.df['talk_id'].duplicated().sum()
        if duplicate_ids > 0:
            self.errors.append(f"{duplicate_ids} duplicate talk_ids")

        print(f"✅ Critical fields validated")

    def validate_data_formats(self):
        """Validate data formats and types"""

        # Check topics format
        topics_sample = self.df['topics'].dropna().head()
        topics_format = None

        for topic in topics_sample:
            try:
                parsed = json.loads(topic)
                topics_format = "JSON"
                break
            except:
                topics_format = "String"

        self.stats['topics_format'] = topics_format
        print(f"📊 Topics format: {topics_format}")

        # Check dates
        date_cols = ['recorded_date', 'published_date']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    pd.to_datetime(self.df[col].dropna().head())
                    print(f"✅ {col} is parseable")
                except:
                    self.warnings.append(f"{col} has invalid date format")

        # Check numeric fields
        numeric_cols = ['views', 'comments', 'duration']
        for col in numeric_cols:
            if col in self.df.columns:
                non_numeric = pd.to_numeric(self.df[col], errors='coerce').isnull().sum()
                if non_numeric > 0:
                    self.warnings.append(f"{col}: {non_numeric} non-numeric values")

    def validate_text_quality(self):
        """Check text field quality"""

        # Transcript length distribution
        transcript_lengths = self.df['transcript'].str.len()

        # Flag suspiciously short transcripts
        very_short = (transcript_lengths < 500).sum()
        if very_short > 0:
            self.warnings.append(f"{very_short} talks with <500 chars transcript")

        # Check for encoding issues
        encoding_issues = 0
        for text in self.df['transcript'].dropna().head(100):
            if '\\u' in text or '\\x' in text or '�' in text:
                encoding_issues += 1

        if encoding_issues > 0:
            self.warnings.append(f"Potential encoding issues in {encoding_issues}/100 samples")

        self.stats['avg_transcript_length'] = transcript_lengths.mean()
        self.stats['min_transcript_length'] = transcript_lengths.min()
        self.stats['max_transcript_length'] = transcript_lengths.max()

    def validate_for_rag(self):
        """RAG-specific validations"""

        # Estimate tokens and costs
        avg_tokens = self.df['transcript'].str.len().mean() / 4  # rough estimate
        total_tokens = self.df['transcript'].str.len().sum() / 4

        self.stats['estimated_total_tokens'] = total_tokens
        self.stats['estimated_embedding_cost'] = (total_tokens / 1000) * 0.00002

        # Check if we have enough diversity
        unique_speakers = self.df['speaker_1'].nunique()
        if unique_speakers < 50:
            self.warnings.append(f"Low speaker diversity: {unique_speakers} unique speakers")

        # Topic coverage
        if self.stats['topics_format'] == 'JSON':
            try:
                all_topics = []
                for topics in self.df['topics'].dropna():
                    all_topics.extend(json.loads(topics))
                unique_topics = len(set(all_topics))
                self.stats['unique_topics'] = unique_topics
            except:
                pass

    def generate_report(self) -> Dict:
        """Generate validation report"""

        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        # Errors (must fix)
        if self.errors:
            print("\n❌ ERRORS (must fix):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ No critical errors")

        # Warnings (should review)
        if self.warnings:
            print("\n⚠️  WARNINGS (review):")
            for warning in self.warnings:
                print(f"  - {warning}")

        # Statistics
        print("\n📊 STATISTICS:")
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:,.2f}")
            else:
                print(f"  - {key}: {value}")

        # RAG readiness
        print("\n" + "=" * 60)
        print("RAG READINESS:")
        print("=" * 60)

        is_ready = len(self.errors) == 0

        if is_ready:
            print("✅ Dataset is ready for RAG pipeline")
            print(f"💰 Estimated embedding cost: ${self.stats.get('estimated_embedding_cost', 0):.4f}")
        else:
            print("❌ Dataset needs fixes before RAG processing")

        return {
            'is_ready': is_ready,
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats
        }

    def save_clean_subset(self, n_talks: int = 100):
        """Save a clean subset for testing"""
        if not self.errors:
            # Select diverse subset
            clean_df = self.df.dropna(subset=['transcript', 'title', 'talk_id'])

            # Sample with diversity
            if len(clean_df) > n_talks:
                sample = clean_df.sample(n=n_talks, random_state=42)
            else:
                sample = clean_df

            sample.to_csv(f'ted_clean_{n_talks}.csv', index=False)
            print(f"\n✅ Saved {len(sample)} clean talks to 'ted_clean_{n_talks}.csv'")

            # Calculate test cost
            test_tokens = sample['transcript'].str.len().sum() / 4
            test_cost = (test_tokens / 1000) * 0.00002
            print(f"💰 Test subset embedding cost: ${test_cost:.4f}")


def main():
    # Run validation
    validator = TEDDataValidator('../data/ted_talks_en.csv')

    if validator.load_data():
        validator.validate_required_columns()
        validator.validate_critical_fields()
        validator.validate_data_formats()
        validator.validate_text_quality()
        validator.validate_for_rag()

        report = validator.generate_report()

        # Save clean subset if ready
        if report['is_ready']:
            validator.save_clean_subset(100)

    return validator


if __name__ == "__main__":
    validator = main()