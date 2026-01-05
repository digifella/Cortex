"""
Workspace Git Integration
Version: 1.0.0
Date: 2026-01-05

Purpose: Git integration for workspace version control.
Automatically commits changes to workspace files with descriptive messages.
"""

import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from .utils import get_logger

logger = get_logger(__name__)


class WorkspaceGit:
    """Git integration for workspace version control."""

    def __init__(self, workspace_path: Path):
        """
        Initialize Git integration for workspace.

        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.git_dir = self.workspace_path / ".git"

        # Initialize git repo if it doesn't exist
        if not self.git_dir.exists():
            self._init_repo()

        logger.info(f"WorkspaceGit initialized for {workspace_path}")

    def _init_repo(self):
        """Initialize Git repository in workspace."""
        try:
            # Initialize repo
            subprocess.run(
                ["git", "init"],
                cwd=self.workspace_path,
                check=True,
                capture_output=True,
                text=True
            )

            # Create .gitignore
            gitignore_path = self.workspace_path / ".gitignore"
            gitignore_path.write_text(
                "# Temporary files\n"
                "*.tmp\n"
                "*.bak\n"
                "~*\n"
                "\n"
                "# Python\n"
                "__pycache__/\n"
                "*.py[cod]\n"
                "\n"
                "# OS files\n"
                ".DS_Store\n"
                "Thumbs.db\n"
            )

            # Create README
            readme_path = self.workspace_path / "README.md"
            readme_path.write_text(
                f"# Workspace: {self.workspace_path.name}\n\n"
                f"Tender response workspace with version-controlled documents.\n\n"
                f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

            # Initial commit
            self._add_all()
            self._commit("Initial commit: Workspace created")

            logger.info(f"Initialized Git repository in {self.workspace_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Git repo: {e}")
            raise

    def _run_git_command(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """
        Run git command in workspace directory.

        Args:
            args: Git command arguments
            check: Whether to raise exception on error

        Returns:
            CompletedProcess result
        """
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.workspace_path,
                check=check,
                capture_output=True,
                text=True
            )
            return result

        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: git {' '.join(args)}")
            logger.error(f"Error: {e.stderr}")
            raise

    def _add_all(self):
        """Add all changes to staging."""
        self._run_git_command(["add", "."])

    def _add_files(self, files: List[str]):
        """Add specific files to staging."""
        self._run_git_command(["add"] + files)

    def _commit(self, message: str, author: Optional[str] = None) -> bool:
        """
        Create a commit with message.

        Args:
            message: Commit message
            author: Optional author string (e.g., "Name <email>")

        Returns:
            True if commit created, False if nothing to commit
        """
        try:
            args = ["commit", "-m", message]

            if author:
                args.extend(["--author", author])

            self._run_git_command(args)
            logger.info(f"Created commit: {message[:50]}...")
            return True

        except subprocess.CalledProcessError as e:
            if "nothing to commit" in e.stderr:
                logger.debug("Nothing to commit")
                return False
            raise

    def commit_changes(
        self,
        message: str,
        files: Optional[List[str]] = None,
        author: Optional[str] = None
    ) -> bool:
        """
        Commit changes to workspace.

        Args:
            message: Commit message
            files: Optional list of specific files to commit (None = all changes)
            author: Optional author string

        Returns:
            True if commit created

        Examples:
            >>> git = WorkspaceGit(workspace_path)
            >>> git.commit_changes("LLM markup: 47 mentions suggested")
            True

            >>> git.commit_changes(
            ...     "Markup reviewed: 42 approved, 5 modified",
            ...     files=["tender_marked_up.docx", "field_bindings.yaml"],
            ...     author="paul.smith@longboardfella.com.au"
            ... )
            True
        """
        if files:
            self._add_files(files)
        else:
            self._add_all()

        return self._commit(message, author)

    def get_status(self) -> Dict[str, List[str]]:
        """
        Get current Git status.

        Returns:
            Dict with 'modified', 'added', 'deleted', 'untracked' lists
        """
        result = self._run_git_command(["status", "--porcelain"])

        status = {
            'modified': [],
            'added': [],
            'deleted': [],
            'untracked': []
        }

        for line in result.stdout.split('\n'):
            if not line:
                continue

            code = line[:2]
            filename = line[3:]

            if code == ' M' or code == 'M ':
                status['modified'].append(filename)
            elif code == 'A ' or code == 'AM':
                status['added'].append(filename)
            elif code == ' D' or code == 'D ':
                status['deleted'].append(filename)
            elif code == '??':
                status['untracked'].append(filename)

        return status

    def get_log(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get commit history.

        Args:
            limit: Number of commits to return

        Returns:
            List of commit dicts with 'hash', 'author', 'date', 'message'

        Examples:
            >>> git = WorkspaceGit(workspace_path)
            >>> commits = git.get_log(limit=5)
            >>> for commit in commits:
            ...     print(f"{commit['date']}: {commit['message']}")
        """
        result = self._run_git_command([
            "log",
            f"-{limit}",
            "--pretty=format:%H|%an|%ai|%s"
        ])

        commits = []

        for line in result.stdout.split('\n'):
            if not line:
                continue

            parts = line.split('|', 3)
            if len(parts) == 4:
                commits.append({
                    'hash': parts[0],
                    'author': parts[1],
                    'date': parts[2],
                    'message': parts[3]
                })

        return commits

    def create_tag(self, tag_name: str, message: Optional[str] = None) -> bool:
        """
        Create an annotated tag.

        Args:
            tag_name: Tag name (e.g., "v1.0", "final_submission")
            message: Optional tag message

        Returns:
            True if tag created

        Examples:
            >>> git = WorkspaceGit(workspace_path)
            >>> git.create_tag("v1.0", "Final submission version")
            True
        """
        try:
            if message:
                self._run_git_command(["tag", "-a", tag_name, "-m", message])
            else:
                self._run_git_command(["tag", tag_name])

            logger.info(f"Created tag: {tag_name}")
            return True

        except subprocess.CalledProcessError:
            logger.warning(f"Failed to create tag: {tag_name}")
            return False

    def checkout_tag(self, tag_name: str) -> bool:
        """
        Checkout a specific tag (read-only).

        Args:
            tag_name: Tag name

        Returns:
            True if checkout successful
        """
        try:
            self._run_git_command(["checkout", tag_name])
            logger.info(f"Checked out tag: {tag_name}")
            return True

        except subprocess.CalledProcessError:
            logger.warning(f"Failed to checkout tag: {tag_name}")
            return False

    def get_diff(self, file_path: Optional[str] = None) -> str:
        """
        Get diff of changes.

        Args:
            file_path: Optional specific file to diff

        Returns:
            Diff output as string
        """
        args = ["diff"]
        if file_path:
            args.append(file_path)

        result = self._run_git_command(args, check=False)
        return result.stdout

    def reset_file(self, file_path: str) -> bool:
        """
        Reset a file to last committed state.

        Args:
            file_path: File to reset

        Returns:
            True if reset successful
        """
        try:
            self._run_git_command(["checkout", "--", file_path])
            logger.info(f"Reset file: {file_path}")
            return True

        except subprocess.CalledProcessError:
            logger.warning(f"Failed to reset file: {file_path}")
            return False

    def has_changes(self) -> bool:
        """
        Check if there are uncommitted changes.

        Returns:
            True if there are changes
        """
        status = self.get_status()
        return any(status.values())


# ============================================
# WORKSPACE GIT OPERATIONS
# ============================================

class WorkspaceGitOperations:
    """High-level Git operations for workspace workflow."""

    @staticmethod
    def on_workspace_created(workspace_path: Path, tender_filename: str) -> WorkspaceGit:
        """
        Initialize Git for new workspace.

        Args:
            workspace_path: Workspace directory
            tender_filename: Name of tender document

        Returns:
            WorkspaceGit instance
        """
        git = WorkspaceGit(workspace_path)
        git.commit_changes(f"Workspace created: Tender {tender_filename} uploaded")
        return git

    @staticmethod
    def on_llm_markup_completed(
        git: WorkspaceGit,
        mention_count: int
    ) -> bool:
        """
        Commit after LLM markup completion.

        Args:
            git: WorkspaceGit instance
            mention_count: Number of mentions suggested

        Returns:
            True if committed
        """
        return git.commit_changes(
            f"LLM markup: {mention_count} mentions suggested",
            files=["tender_marked_up.docx", "field_bindings.yaml"]
        )

    @staticmethod
    def on_human_review_completed(
        git: WorkspaceGit,
        approved_count: int,
        modified_count: int,
        reviewer_email: str
    ) -> bool:
        """
        Commit after human review.

        Args:
            git: WorkspaceGit instance
            approved_count: Number of mentions approved
            modified_count: Number of mentions modified
            reviewer_email: Reviewer's email

        Returns:
            True if committed
        """
        return git.commit_changes(
            f"Markup reviewed: {approved_count} approved, {modified_count} modified",
            files=["tender_marked_up.docx", "field_bindings.yaml"],
            author=f"{reviewer_email} <{reviewer_email}>"
        )

    @staticmethod
    def on_entity_bound(
        git: WorkspaceGit,
        entity_name: str
    ) -> bool:
        """
        Commit after entity profile bound.

        Args:
            git: WorkspaceGit instance
            entity_name: Entity name

        Returns:
            True if committed
        """
        return git.commit_changes(
            f"Entity bound: {entity_name}",
            files=["metadata.yaml", "field_bindings.yaml"]
        )

    @staticmethod
    def on_content_generated(
        git: WorkspaceGit,
        section_count: int
    ) -> bool:
        """
        Commit after content generation.

        Args:
            git: WorkspaceGit instance
            section_count: Number of sections generated

        Returns:
            True if committed
        """
        return git.commit_changes(
            f"Content generated: {section_count} sections filled",
            files=["tender_filled.docx", "generation_log.json"]
        )

    @staticmethod
    def on_reviewer_feedback(
        git: WorkspaceGit,
        section_name: str,
        feedback: str,
        reviewer_email: str
    ) -> bool:
        """
        Commit after reviewer provides feedback.

        Args:
            git: WorkspaceGit instance
            section_name: Section with feedback
            feedback: Feedback summary
            reviewer_email: Reviewer email

        Returns:
            True if committed
        """
        return git.commit_changes(
            f"Review feedback: {section_name} - {feedback}",
            files=["approval_status.yaml"],
            author=f"{reviewer_email} <{reviewer_email}>"
        )

    @staticmethod
    def on_revision_completed(
        git: WorkspaceGit,
        section_name: str,
        author_email: str
    ) -> bool:
        """
        Commit after revision.

        Args:
            git: WorkspaceGit instance
            section_name: Revised section
            author_email: Author email

        Returns:
            True if committed
        """
        return git.commit_changes(
            f"Revision: {section_name} updated",
            files=["tender_filled.docx"],
            author=f"{author_email} <{author_email}>"
        )

    @staticmethod
    def on_final_approval(
        git: WorkspaceGit,
        approver_email: str
    ) -> bool:
        """
        Commit and tag after final approval.

        Args:
            git: WorkspaceGit instance
            approver_email: Approver email

        Returns:
            True if committed
        """
        committed = git.commit_changes(
            "Final approval granted",
            files=["approval_status.yaml"],
            author=f"{approver_email} <{approver_email}>"
        )

        if committed:
            git.create_tag("approved", "Final approval granted")

        return committed

    @staticmethod
    def on_final_export(
        git: WorkspaceGit,
        version: str = "1.0"
    ) -> bool:
        """
        Commit and tag final export.

        Args:
            git: WorkspaceGit instance
            version: Version number

        Returns:
            True if committed
        """
        committed = git.commit_changes(
            f"Final export: v{version} ready for submission",
            files=["tender_final.docx"]
        )

        if committed:
            git.create_tag(f"v{version}", f"Final submission version {version}")

        return committed
