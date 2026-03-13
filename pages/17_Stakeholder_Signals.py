from __future__ import annotations

import time
from typing import List

import streamlit as st

from cortex_engine.stakeholder_signal_store import StakeholderSignalStore
from cortex_engine.version_config import VERSION_STRING


st.set_page_config(page_title="Stakeholder Signals", layout="wide", page_icon="🎯")


TARGET_TYPES = ["person", "organisation", "industry", "theme"]
PROFILE_STATUSES = ["active", "watch", "inactive", "former"]
WATCH_STATUSES = ["off", "watch"]


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _format_matches(matches: list[dict]) -> str:
    if not matches:
        return "No matches"
    return " | ".join(f"{item['canonical_name']} ({item['score']:.2f})" for item in matches[:3])


def _address_field(address: dict, key: str) -> str:
    if not isinstance(address, dict):
        return ""
    return str(address.get(key) or "").strip()


def main():
    st.title("Stakeholder Signals")
    st.caption(f"Cortex {VERSION_STRING}")
    st.markdown(
        "Review forwarded stakeholder signals, inspect top profile matches, and identify low-confidence items that need analyst review."
    )

    store = StakeholderSignalStore()
    state = store.get_state()
    profiles = list(state.get("profiles") or [])
    org_options = sorted({profile.get("org_name", "") for profile in profiles if profile.get("org_name")})
    org_filter = st.selectbox("Organisation", ["All"] + org_options, index=0)

    top_left, top_mid, top_right = st.columns([2, 1, 1])
    with top_left:
        st.code(f"Store: {store.state_path}")
    with top_mid:
        matched_only = st.toggle("Matched only", value=False)
    with top_right:
        auto_refresh = st.toggle("Auto refresh (5s)", value=False)

    if auto_refresh:
        time.sleep(5)
        st.rerun()

    selected_org = "" if org_filter == "All" else org_filter
    signals = store.list_signals(org_name=selected_org, matched_only=matched_only, limit=200)
    filtered_profiles = store.list_profiles(org_name=selected_org)
    suggestions = store.list_update_suggestions(org_name=selected_org, limit=200)
    observed_facts = store.list_observed_facts(org_name=selected_org, limit=200)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signals", len(signals))
    col2.metric("Profiles", len(filtered_profiles))
    col3.metric("Matched", sum(1 for signal in signals if signal.get("matches")))
    col4.metric("Needs Review", sum(1 for signal in signals if signal.get("needs_review")) + sum(1 for item in suggestions if item.get("status") == "pending"))

    st.subheader("Manual Testing")
    tab_profiles, tab_signals, tab_digest = st.tabs(["Add Target", "Add Signal", "Generate Digest"])

    with tab_profiles:
        with st.form("stakeholder_profile_form", clear_on_submit=True):
            pf_col1, pf_col2 = st.columns(2)
            with pf_col1:
                form_org_name = st.text_input("Organisation", value=selected_org or "")
                target_type = st.selectbox("Target Type", TARGET_TYPES, index=0)
                canonical_name = st.text_input("Canonical Name")
                external_profile_id = st.text_input("External Profile ID")
                email = st.text_input("Email")
                function_name = st.text_input("Function")
                industry = st.text_input("Industry")
                status = st.selectbox("Status", PROFILE_STATUSES, index=0)
                watch_status = st.selectbox("Watch Status", WATCH_STATUSES, index=0)
                last_verified_at = st.text_input("Last Verified At", value="")
                linkedin_url = st.text_input("LinkedIn URL")
                current_employer = st.text_input("Current Employer / Org")
                website_url = st.text_input("Website URL")
                parent_entity = st.text_input("Parent Entity")
            with pf_col2:
                current_role = st.text_input("Current Role")
                acn_abn = st.text_input("ACN / ABN")
                phone = st.text_input("Phone")
                aliases_csv = st.text_input("Aliases (comma separated)")
                employers_csv = st.text_input("Known Employers (comma separated)")
                tags_csv = st.text_input("Tags (comma separated)")
                notes = st.text_area("Notes", height=120)
                address_street = st.text_input("Address Street")
                address_city = st.text_input("Address City")
                address_state = st.text_input("Address State")
                address_postcode = st.text_input("Address Postcode")
                address_country = st.text_input("Address Country", value="Australia")
            submit_profile = st.form_submit_button("Save Target", use_container_width=True)
            if submit_profile:
                if not form_org_name.strip() or not canonical_name.strip():
                    st.error("Organisation and Canonical Name are required.")
                else:
                    result = store.upsert_profiles(
                        org_name=form_org_name.strip(),
                        profiles=[
                            {
                                "canonical_name": canonical_name.strip(),
                                "target_type": target_type,
                                "external_profile_id": external_profile_id.strip(),
                                "email": email.strip(),
                                "function": function_name.strip(),
                                "industry": industry.strip(),
                                "status": status,
                                "watch_status": watch_status,
                                "last_verified_at": last_verified_at.strip(),
                                "linkedin_url": linkedin_url.strip(),
                                "current_employer": current_employer.strip(),
                                "current_role": current_role.strip(),
                                "website_url": website_url.strip(),
                                "parent_entity": parent_entity.strip(),
                                "acn_abn": acn_abn.strip(),
                                "phone": phone.strip(),
                                "address": {
                                    "street": address_street.strip(),
                                    "city": address_city.strip(),
                                    "state": address_state.strip(),
                                    "postcode": address_postcode.strip(),
                                    "country": address_country.strip(),
                                },
                                "aliases": _split_csv(aliases_csv),
                                "known_employers": _split_csv(employers_csv),
                                "tags": _split_csv(tags_csv),
                                "notes": notes.strip(),
                            }
                        ],
                        source="streamlit_manual",
                    )
                    st.success(
                        f"Saved target for {form_org_name.strip()} ({result['added']} added, {result['updated']} updated)."
                    )
                    st.rerun()

    with tab_signals:
        with st.form("signal_ingest_form", clear_on_submit=True):
            sg_col1, sg_col2 = st.columns(2)
            with sg_col1:
                signal_org_name = st.text_input("Organisation ", value=selected_org or "", key="signal_org_name")
                signal_target_type = st.selectbox("Signal Target Type", TARGET_TYPES, index=0)
                subject = st.text_input("Subject")
                primary_url = st.text_input("Primary URL")
                candidate_name = st.text_input("Candidate Name")
            with sg_col2:
                candidate_employer = st.text_input("Candidate Employer")
                tags = st.text_input("Signal Tags (comma separated)")
                notification_kind = st.text_input("Notification Kind", value="manual_test")
                text_note = st.text_area("Analyst Note", height=100)
            raw_text = st.text_area("Raw Signal Text", height=220)
            submit_signal = st.form_submit_button("Ingest Signal", use_container_width=True)
            if submit_signal:
                if not signal_org_name.strip() or not (subject.strip() or raw_text.strip()):
                    st.error("Organisation and either Subject or Raw Signal Text are required.")
                else:
                    signal = store.ingest_signal(
                        {
                            "org_name": signal_org_name.strip(),
                            "source_system": "streamlit_manual",
                            "signal_type": "manual_market_intel",
                            "target_type": signal_target_type,
                            "subject": subject.strip(),
                            "raw_text": raw_text.strip(),
                            "primary_url": primary_url.strip(),
                            "text_note": text_note.strip(),
                            "parsed_candidate_name": candidate_name.strip(),
                            "parsed_candidate_employer": candidate_employer.strip(),
                            "notification_kind": notification_kind.strip(),
                            "tags": _split_csv(tags),
                        }
                    )
                    st.success(f"Ingested {signal['signal_id']} with {len(signal.get('matches') or [])} match(es).")
                    if signal.get("matches"):
                        st.json(signal["matches"])
                    st.rerun()

    with tab_digest:
        with st.form("signal_digest_form", clear_on_submit=True):
            dg_col1, dg_col2 = st.columns(2)
            with dg_col1:
                digest_org_name = st.text_input("Organisation  ", value=selected_org or "", key="digest_org_name")
                since_ts = st.text_input("Since Timestamp (optional)", value="")
                max_items = st.number_input("Max Items", min_value=1, max_value=200, value=25, step=1)
            with dg_col2:
                include_needs_review = st.checkbox("Include Needs Review", value=True)
                matched_only_digest = st.checkbox("Matched Only", value=True)
                llm_synthesis = st.checkbox("LLM Synthesis", value=False)
                llm_provider = st.selectbox("LLM Provider", ["ollama", "anthropic"], index=0)
                llm_model = st.text_input("LLM Model", value="qwen3.5:9b")
                profile_key_filter = st.text_input("Profile Keys Filter (comma separated)")
            submit_digest = st.form_submit_button("Generate Digest", use_container_width=True)
            if submit_digest:
                if not digest_org_name.strip():
                    st.error("Organisation is required.")
                else:
                    digest = store.generate_digest(
                        org_name=digest_org_name.strip(),
                        since_ts=since_ts.strip(),
                        profile_keys=_split_csv(profile_key_filter),
                        max_items=int(max_items),
                        include_needs_review=include_needs_review,
                        matched_only=matched_only_digest,
                        llm_synthesis=llm_synthesis,
                        llm_provider=llm_provider,
                        llm_model=llm_model.strip(),
                    )
                    st.success(f"Generated digest {digest['digest_id']} with {digest['signal_count']} signal(s).")
                    st.caption(
                        f"LLM synthesised: {digest.get('llm_synthesised', False)} | "
                        f"Profiles covered: {digest.get('profiles_covered', 0)} | "
                        f"Period: {digest.get('period_start', '')} -> {digest.get('period_end', '')}"
                    )
                    digest_text = open(digest["output_path"], "r", encoding="utf-8").read()
                    st.download_button(
                        "Download Digest Markdown",
                        data=digest_text,
                        file_name=f"{digest['digest_id']}.md",
                        mime="text/markdown",
                    )
                    st.code(digest_text)

    with st.expander("Stakeholder Profiles", expanded=False):
        if not filtered_profiles:
            st.info("No stakeholder profiles stored yet.")
        else:
            for profile in filtered_profiles:
                st.json(profile)

    st.subheader("Registry Manager")
    if not filtered_profiles:
        st.info("No target profiles available to edit.")
    else:
        profile_labels = {
            profile["profile_key"]: f"{profile.get('canonical_name', '')} | {profile.get('target_type', '')} | {profile.get('org_name', '')}"
            for profile in filtered_profiles
        }
        selected_profile_key = st.selectbox(
            "Select Target Profile",
            options=list(profile_labels.keys()),
            format_func=lambda key: profile_labels[key],
        )
        selected_profile = store.get_profile(selected_profile_key) or {}
        with st.form("edit_profile_form", clear_on_submit=False):
            rp_col1, rp_col2 = st.columns(2)
            with rp_col1:
                edit_name = st.text_input("Canonical Name", value=selected_profile.get("canonical_name", ""))
                edit_target_type = st.selectbox(
                    "Target Type",
                    TARGET_TYPES,
                    index=max(0, TARGET_TYPES.index(selected_profile.get("target_type", "person")))
                    if selected_profile.get("target_type", "person") in TARGET_TYPES else 0,
                    key="edit_target_type",
                )
                edit_org = st.text_input("Organisation Scope", value=selected_profile.get("org_name", ""))
                edit_external_profile_id = st.text_input(
                    "External Profile ID",
                    value=selected_profile.get("external_profile_id", ""),
                    disabled=True,
                )
                edit_email = st.text_input("Email", value=selected_profile.get("email", ""))
                edit_function = st.text_input("Function", value=selected_profile.get("function", ""))
                edit_industry = st.text_input("Industry", value=selected_profile.get("industry", ""))
                edit_status = st.selectbox(
                    "Status",
                    PROFILE_STATUSES,
                    index=max(0, PROFILE_STATUSES.index(selected_profile.get("status", "active")))
                    if selected_profile.get("status", "active") in PROFILE_STATUSES else 0,
                    key="edit_profile_status",
                )
                edit_watch_status = st.selectbox(
                    "Watch Status",
                    WATCH_STATUSES,
                    index=max(0, WATCH_STATUSES.index(selected_profile.get("watch_status", "off")))
                    if selected_profile.get("watch_status", "off") in WATCH_STATUSES else 0,
                    key="edit_profile_watch_status",
                )
                edit_last_verified_at = st.text_input("Last Verified At", value=selected_profile.get("last_verified_at", ""))
                edit_linkedin = st.text_input("LinkedIn URL", value=selected_profile.get("linkedin_url", ""))
                edit_employer = st.text_input("Current Employer / Org", value=selected_profile.get("current_employer", ""))
                edit_website = st.text_input("Website URL", value=selected_profile.get("website_url", ""))
                edit_parent = st.text_input("Parent Entity", value=selected_profile.get("parent_entity", ""))
            with rp_col2:
                edit_role = st.text_input("Current Role", value=selected_profile.get("current_role", ""))
                edit_acn_abn = st.text_input("ACN / ABN", value=selected_profile.get("acn_abn", ""))
                edit_phone = st.text_input("Phone", value=selected_profile.get("phone", ""))
                edit_aliases = st.text_input("Aliases", value=", ".join(selected_profile.get("aliases") or []))
                edit_employers = st.text_input("Known Employers", value=", ".join(selected_profile.get("known_employers") or []))
                edit_tags = st.text_input("Tags", value=", ".join(selected_profile.get("tags") or []))
                edit_notes = st.text_area("Notes", value=selected_profile.get("notes", ""), height=120)
                edit_address_street = st.text_input("Address Street", value=_address_field(selected_profile.get("address", {}), "street"))
                edit_address_city = st.text_input("Address City", value=_address_field(selected_profile.get("address", {}), "city"))
                edit_address_state = st.text_input("Address State", value=_address_field(selected_profile.get("address", {}), "state"))
                edit_address_postcode = st.text_input("Address Postcode", value=_address_field(selected_profile.get("address", {}), "postcode"))
                edit_address_country = st.text_input("Address Country", value=_address_field(selected_profile.get("address", {}), "country"))
            e1, e2 = st.columns(2)
            with e1:
                save_profile = st.form_submit_button("Update Target Profile", use_container_width=True)
            with e2:
                delete_profile = st.form_submit_button("Delete Target Profile", use_container_width=True)

            if save_profile:
                updated = store.save_profile(
                    selected_profile_key,
                    {
                        "canonical_name": edit_name.strip(),
                        "target_type": edit_target_type,
                        "org_name": edit_org.strip(),
                        "external_profile_id": edit_external_profile_id.strip(),
                        "email": edit_email.strip(),
                        "function": edit_function.strip(),
                        "industry": edit_industry.strip(),
                        "status": edit_status,
                        "watch_status": edit_watch_status,
                        "last_verified_at": edit_last_verified_at.strip(),
                        "linkedin_url": edit_linkedin.strip(),
                        "current_employer": edit_employer.strip(),
                        "current_role": edit_role.strip(),
                        "website_url": edit_website.strip(),
                        "parent_entity": edit_parent.strip(),
                        "acn_abn": edit_acn_abn.strip(),
                        "phone": edit_phone.strip(),
                        "address": {
                            "street": edit_address_street.strip(),
                            "city": edit_address_city.strip(),
                            "state": edit_address_state.strip(),
                            "postcode": edit_address_postcode.strip(),
                            "country": edit_address_country.strip(),
                        },
                        "aliases": _split_csv(edit_aliases),
                        "known_employers": _split_csv(edit_employers),
                        "tags": _split_csv(edit_tags),
                        "notes": edit_notes.strip(),
                    },
                )
                if updated:
                    st.success("Target profile updated.")
                    st.rerun()
                st.error("Could not update target profile.")
            if delete_profile:
                if store.delete_profile(selected_profile_key):
                    st.success("Target profile deleted.")
                    st.rerun()
                st.error("Could not delete target profile.")

    st.subheader("Pending Target Updates")
    pending_suggestions = [item for item in suggestions if item.get("status") == "pending"]
    if not suggestions:
        st.info("No target update suggestions yet.")
    else:
        for suggestion in suggestions:
            title = (
                f"{suggestion.get('canonical_name', '')} | {suggestion.get('field_name', '')} | "
                f"{suggestion.get('old_value', '—')} -> {suggestion.get('proposed_value', '')} | "
                f"{suggestion.get('status', 'pending')}"
            )
            with st.expander(title, expanded=suggestion in pending_suggestions):
                st.write(
                    f"**Org:** {suggestion.get('org_name', '')}  \n"
                    f"**Signal ID:** `{suggestion.get('signal_id', '')}`  \n"
                    f"**Confidence:** {suggestion.get('confidence', 0):.2f}  \n"
                    f"**Evidence:** {suggestion.get('evidence_excerpt', '')}"
                )
                a1, a2, a3 = st.columns(3)
                with a1:
                    if st.button("Accept", key=f"sug_accept_{suggestion['suggestion_id']}", use_container_width=True):
                        store.review_update_suggestion(
                            suggestion["suggestion_id"],
                            "accepted",
                            apply_to_profile=True,
                        )
                        st.success("Suggestion marked accepted.")
                        st.rerun()
                with a2:
                    if st.button("Reject", key=f"sug_reject_{suggestion['suggestion_id']}", use_container_width=True):
                        store.review_update_suggestion(suggestion["suggestion_id"], "rejected")
                        st.success("Suggestion marked rejected.")
                        st.rerun()
                with a3:
                    if st.button("Reset", key=f"sug_reset_{suggestion['suggestion_id']}", use_container_width=True):
                        store.review_update_suggestion(suggestion["suggestion_id"], "pending")
                        st.success("Suggestion reset to pending.")
                        st.rerun()
                st.json(suggestion)

    with st.expander("Observed Facts", expanded=False):
        if not observed_facts:
            st.info("No observed facts recorded yet.")
        else:
            for fact in observed_facts:
                st.json(fact)

    st.subheader("Recent Signals")
    if not signals:
        st.info("No signals stored yet.")
        return

    for signal in signals:
        title = f"{signal['received_at']} | {signal['signal_id']} | {signal.get('subject') or signal.get('signal_type')}"
        with st.expander(title, expanded=bool(signal.get("needs_review"))):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write(
                    f"**Org:** {signal.get('org_name', '')}  \n"
                    f"**Target Type:** {signal.get('target_type', '')}  \n"
                    f"**Candidate:** {signal.get('parsed_candidate_name', '')}  \n"
                    f"**Employer:** {signal.get('parsed_candidate_employer', '')}  \n"
                    f"**Primary URL:** {signal.get('primary_url', '') or '—'}  \n"
                    f"**Top Matches:** {_format_matches(signal.get('matches') or [])}"
                )
                if signal.get("text_note"):
                    st.markdown("**Note**")
                    st.write(signal["text_note"])
                if signal.get("raw_text"):
                    st.markdown("**Raw Text**")
                    st.code(signal["raw_text"][:4000])
            with c2:
                st.metric("Matches", len(signal.get("matches") or []))
                st.metric("Needs Review", "Yes" if signal.get("needs_review") else "No")
                st.metric("Tags", len(signal.get("tags") or []))
                st.metric("Updates", len(signal.get("update_suggestion_ids") or []))
                if st.button("Delete Signal", key=f"delete_signal_{signal['signal_id']}", use_container_width=True):
                    if store.delete_signal(signal["signal_id"]):
                        st.success("Signal deleted.")
                        st.rerun()
                    st.error("Could not delete signal.")
            st.json(signal)


if __name__ == "__main__":
    main()
