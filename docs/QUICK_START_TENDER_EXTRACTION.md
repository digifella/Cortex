# Quick Start: Tender Data Extraction (Phase 1)

## 🎯 What We Built

A **data extraction system** that analyzes your knowledge base and extracts structured organizational data for auto-filling tender documents.

**Key Insight:** Tender completion is 95% data extraction, not creative generation.

---

## 🚀 How to Test (5 Minutes)

### Step 1: Ensure You Have KB Documents

Your knowledge base should contain documents with:
- Company registration details (ABN, ACN, address)
- Insurance certificates (policy numbers, coverage, expiry)
- Team CVs/resumes (qualifications, experience)
- Project case studies (clients, deliverables, outcomes)
- Reference letters

**If you don't have these yet:**
```bash
# Go to Knowledge Ingest page and add some company documents
streamlit run Cortex_Suite.py
# Navigate to: 2. Knowledge Ingest
# Upload your company docs
```

### Step 2: Run Structured Data Extraction

1. **Start the application:**
   ```bash
   streamlit run Cortex_Suite.py
   ```

2. **Navigate to Knowledge Search:**
   - Click "3. Knowledge Search" in sidebar

3. **Extract Structured Data:**
   - Look for "📊 Structured Data Extraction" section
   - Click "🚀 Extract Structured Data" button
   - Watch progress (2-5 minutes depending on KB size)

4. **View Results:**
   - See extraction summary with counts
   - Check metrics: Insurance Policies, Qualifications, Projects, References

### Step 3: Inspect Extracted Data

**File Location:**
```
{your_db_path}/structured_knowledge.json
```

**Example path:**
```
/mnt/f/ai_databases/structured_knowledge.json
```

**View the file:**
```bash
cat /mnt/f/ai_databases/structured_knowledge.json | jq .
# Or open in any text editor
```

**What You Should See:**
```json
{
  "organization": {
    "legal_name": "Your Company Name",
    "abn": "12345678901",
    "address": {...}
  },
  "insurances": [...],
  "team_qualifications": [...],
  "projects": [...]
}
```

---

## 🧪 Test Scenarios

### Scenario 1: Organization Profile Extraction

**What to check:**
- Legal name extracted correctly
- ABN/ACN extracted (if in KB)
- Address components (street, city, state, postcode)
- Contact details (phone, email)

**If missing:**
- Add company registration document to KB
- Add letterhead PDFs with address
- Re-run extraction

### Scenario 2: Insurance Policies

**What to check:**
- Insurance types identified (Public Liability, Professional Indemnity, etc.)
- Policy numbers extracted
- Coverage amounts extracted
- Expiry dates extracted and formatted correctly

**Validation:**
- Check `is_expired` field - should be `false` for active policies
- Check `days_until_expiry` - should show days remaining

**If missing:**
- Add insurance certificates to KB
- Ensure certificates have policy numbers and dates visible
- Re-run extraction

### Scenario 3: Team Qualifications

**What to check:**
- Person names extracted
- Qualification names extracted
- Institutions/certifying bodies identified
- Dates obtained

**Examples:**
- "Doctor of Philosophy (Health Informatics), University of Sydney, 2018"
- "Certified Health Informatician Australasia (CHIA), HISA, 2020"

**If missing:**
- Add team CVs/resumes to KB
- Add certificates/transcripts
- Re-run extraction

### Scenario 4: Project Experience

**What to check:**
- Project names
- Client organizations
- Dates (start/end)
- Deliverables and outcomes
- Technologies used

**Validation:**
- Check `is_ongoing` for current projects
- Check `duration_months` calculation

**If missing:**
- Add case studies to KB
- Add project reports/summaries
- Re-run extraction

---

## 🔍 Troubleshooting

### Problem: Extraction Returns Empty Results

**Possible Causes:**
1. Knowledge base is empty or has no relevant documents
2. Documents don't contain structured data (e.g., just meeting notes)
3. LLM failed to extract (check logs)

**Solutions:**
1. Verify KB has documents: Check Knowledge Search returns results
2. Add company/organizational documents with key data
3. Check Streamlit console for error messages
4. Manually inspect `structured_knowledge.json` for partial extraction

### Problem: Some Fields Missing

**Possible Causes:**
1. Data not in KB documents
2. Data in image/scanned PDF (OCR issues)
3. LLM extraction missed it

**Solutions:**
1. Add missing source documents to KB
2. Convert scanned PDFs to text-based PDFs
3. Manually edit `structured_knowledge.json` to add missing data
4. Re-run extraction after KB improvements

### Problem: Incorrect Data Extracted

**Possible Causes:**
1. Multiple organizations in KB (extracted wrong one)
2. LLM hallucination
3. Ambiguous data in source documents

**Solutions:**
1. Check `source_documents` field to see where data came from
2. Manually correct `structured_knowledge.json`
3. Improve source document clarity
4. Re-run extraction

### Problem: Extraction is Slow (>10 minutes)

**Possible Causes:**
1. Very large knowledge base (>1000 documents)
2. Slow LLM model
3. Vector search slow

**Solutions:**
1. Normal for large KBs - be patient
2. Check LLM model speed (should use balanced models like llama3.3:70b)
3. Consider upgrading hardware/GPU

---

## 📊 Understanding Extraction Quality

### High Quality Extraction (85-95% accurate)
- Source documents are structured (certificates, official forms)
- Clear labeling ("ABN:", "Policy Number:", etc.)
- Text-based PDFs (not scanned images)

**Example:**
```
Insurance Certificate
Insurer: Insurance Australia Group
Policy Number: PI-2024-12345
Coverage: $20,000,000
Expiry: 30 June 2025
```

### Medium Quality Extraction (70-85% accurate)
- Source documents are semi-structured (CVs, reports)
- Some labeling but inconsistent
- Mixed text and images

**Example:**
```
Jane Smith has a PhD in Health Informatics from
University of Sydney (2018) and is a certified CHIA.
```

### Low Quality Extraction (50-70% accurate)
- Unstructured text (meeting notes, emails)
- No clear labeling
- Scanned images/PDFs

**Example:**
```
We discussed the project with ABC Corp last year.
Jane mentioned her qualifications. The insurance
policy expires soon, need to check.
```

**Recommendation:** For best results, add structured source documents to KB.

---

## ✅ Success Criteria

After running extraction, you should have:

1. **Organization Profile:**
   - [ ] Legal name extracted
   - [ ] ABN/ACN extracted (if applicable)
   - [ ] Address extracted with components
   - [ ] Contact details (phone/email)

2. **Insurance Policies (1+ policies):**
   - [ ] Insurance type identified
   - [ ] Policy number extracted
   - [ ] Expiry date extracted
   - [ ] Coverage amount extracted

3. **Team Qualifications (3+ qualifications):**
   - [ ] Person names matched to qualifications
   - [ ] Qualification names clear
   - [ ] Institutions identified
   - [ ] Dates present

4. **Project Experience (2+ projects):**
   - [ ] Project names extracted
   - [ ] Client organizations identified
   - [ ] Deliverables/outcomes captured

5. **References (1+ references):**
   - [ ] Contact names extracted
   - [ ] Organizations identified
   - [ ] Contact details present

**If you meet 3 of 5 categories, Phase 1 is successful!**

---

## 🔄 Re-Running Extraction

### When to Re-Extract:
- After adding new documents to KB
- After updating insurance policies (renewal)
- After team member qualifications change
- If extraction is >30 days old (system will warn)

### How to Re-Extract:
1. Go to Knowledge Search page
2. Click "🚀 Extract Structured Data" button
3. Wait for completion
4. Old data is overwritten with new extraction

**Note:** Previous `structured_knowledge.json` is replaced, not merged. If you manually edited it, back it up first.

---

## 📝 Next Steps After Phase 1

Once extraction works well:

### Phase 2: Field Classification & Matching (2-3 weeks)
- Upload a real tender document (e.g., RFT12493)
- System auto-detects fields that need filling
- Matches tender fields to extracted structured data
- Shows confidence scores (high/medium/low)

**Goal:** See tender fields automatically filled from structured data

### Phase 3: Auto-Fill Workflow UI (2-3 weeks)
- Review/approve workflow
- Bulk approve high-confidence matches
- Manual review medium-confidence matches
- MoE generation for narrative sections (5% of work)
- Export completed tender document

**Goal:** Complete RFT12493 in <30 minutes (vs 2-3 hours manual)

---

## 🆘 Need Help?

**Check Logs:**
```bash
# Streamlit console shows extraction progress
# Look for errors like "Failed to extract..."
```

**Inspect Extraction File:**
```bash
cat {your_db_path}/structured_knowledge.json | jq '.summary_stats'
```

**Common Issues:**
1. **Empty extraction** → Add more source documents to KB
2. **Partial extraction** → Add missing document types (insurance certs, CVs, etc.)
3. **Wrong data** → Check source documents, manually correct JSON
4. **Slow extraction** → Normal for large KB, be patient

**Report Issues:**
- Include extraction summary (number of items found)
- Include error messages from console
- Include example of what's missing vs what you expected

---

**Status:** Phase 1 Complete (2026-01-03)
**Ready to Test:** Yes - Click "Extract Structured Data" in Knowledge Search!
