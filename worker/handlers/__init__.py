"""
Queue Worker Handler Registry

Maps job type strings to handler functions.
Each handler must accept (input_path, input_data, job) and return
{'output_data': dict, 'output_file': Path | None}.
"""

from .pdf_anonymise import handle as pdf_anonymise_handle
from .cortex_sync import handle as cortex_sync_handle
from .pdf_textify import handle as pdf_textify_handle
from .url_ingest import handle as url_ingest_handle
from .youtube_summarise import handle as youtube_summarise_handle
from .signal_episode import handle as signal_episode_handle
from .intel_extract import handle as intel_extract_handle
from .stakeholder_profile_sync import handle as stakeholder_profile_sync_handle
from .signal_ingest import handle as signal_ingest_handle
from .signal_digest import handle as signal_digest_handle

HANDLERS = {
    'pdf_anonymise':   pdf_anonymise_handle,
    'cortex_sync':     cortex_sync_handle,
    'pdf_textify':     pdf_textify_handle,
    'url_ingest':      url_ingest_handle,
    'youtube_summarise': youtube_summarise_handle,
    'signal_episode':  signal_episode_handle,
    'intel_extract': intel_extract_handle,
    'stakeholder_profile_sync': stakeholder_profile_sync_handle,
    'signal_ingest': signal_ingest_handle,
    'signal_digest': signal_digest_handle,
}
