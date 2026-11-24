# Source Permission Assessment — IMSDb (imsdb.com)

*Assessment date: 2026-11-17. Assessed by the automated `inspect-source` command plus a manual review.
Not legal advice. Re-run `python -m src.main inspect-source` to refresh the machine-readable evidence.*

## 1. Summary

| Question | Finding |
|---|---|
| Does `robots.txt` exist? | **No.** `https://imsdb.com/robots.txt` returns **HTTP 404**. Under RFC 9309 a missing robots.txt places no crawl restrictions, but it is **not** a grant of rights. |
| Are there Terms of Service or scraping rules? | **None found.** The footer links only to *Submit*, *Link to IMSDb*, *Disclaimer*, *Privacy policy* (redirect) and *Contact*. No page addresses automated access, bulk download, or reuse. |
| What does the site claim about rights? | The **Disclaimer** page (`/disclaimer.html`) quotes **17 U.S.C. §107 (fair use)** and states: *"All scripts on this site are educational purposes only. None of the scripts are/or will be sold by IMSDb.com."* |
| Who owns the screenplays? | **The respective studios and writers.** IMSDb is not the copyright holder and **cannot license** the screenplays. It hosts them on a fair-use claim. |
| Is there an API or bulk-data offer? | **No.** |
| Is there authentication, a paywall, or anti-bot protection? | No authentication or paywall for listing, detail, or script pages. The comment form uses reCAPTCHA; the crawler **never submits forms**. |
| HTTP caching headers | Responses carry no `ETag` or `Last-Modified`. Change detection therefore uses SHA-256 content hashes. |

## 2. Conclusion: bulk-collection authorization is **NOT verified**

No document from IMSDb or the rights holders authorizes bulk collection. Fair use is a defence that
depends on the user's purpose, not a licence. Whether bulk ingestion for an educational chatbot is fair use
is a legal question that this software cannot decide.

Because of this, the system ships with the following default permission record for `imsdb`:

| Field | Default |
|---|---|
| `level` | **`sample`** — discovery of listing pages plus at most **10** movie records (detail + screenplay), counted **cumulatively** across all runs |
| `crawl --all` | **Refused** (exit code 4) until an operator records `bulk` authorization with evidence |
| Request rate | One request every **5 s** per host (configurable, never below 1 s), at most 2 workers, honest `User-Agent` |
| Forms, comments, `writer.php` pages, TV transcripts | Not crawled |

## 3. How to raise the permission level (operator action)

Bulk collection requires evidence that you are authorized. For example, written permission from IMSDb
and/or the rights holders, or a documented institutional legal review concluding that your use is
permissible. Record that evidence:

```bash
python -m src.main permit --level bulk \
    --granted-by "Jane Doe, University Legal Office" \
    --evidence "Email from IMSDb admin dated 2026-12-01, archived at docs/evidence/imsdb-permission.pdf" \
    --confirm
```

Every change is appended to `source_permissions.history`; nothing is overwritten. You can also *lower* the level
(`--level discovery_only` or `--level denied`), and the change takes effect on the next request.

## 4. Usage restrictions recorded with every record

* **Attribution:** "Screenplay text retrieved from The Internet Movie Script Database (IMSDb), https://imsdb.com. All rights
  belong to the respective copyright holders (studios and writers)."
* **Use restriction:** educational, scholarship, research, and criticism uses only (per the source's fair-use
  disclaimer). Do not sell or redistribute.
* **Storage:** local only. Exports and backups stay on local disk and carry the attribution notice.
* **Personal data:** user-comment author names on detail pages are **not** extracted into structured fields.
  They remain only inside the preserved raw HTML.

## 5. Evidence captured by `inspect-source`

`inspect-source` fetches `robots.txt`, `/disclaimer.html`, `/contact.html`, `/privacy`, and the home page. It stores
each response in the raw archive (`data/raw/…`) with its SHA-256 hash and writes the result to
`source_permissions.policy_documents`. If the disclaimer text changes, a new content hash appears in the evidence
history and the assessment should be repeated.

## 6. Live collection performed during development

* About 15 manual probe requests to learn the page structure (home page, policy pages, two listing pages, two detail pages,
  one script page), spaced 3–4 s apart.
* One discovery run (48 listing pages) and one sample ingestion of 3 records, within the `sample` cap (see the README, "Step 10 results").
* **2026-11-23:** the project owner recorded `bulk` authorization (permission record version 2: "Project owner
  confirmed on 2026-11-23 that approval for full IMSDb collection has been received; approval document not
  attached to this record"). A full crawl (`crawl --all`, job `job-20261123T122924-e837a5`) then ran from 12:29 to
  15:59 UTC at the default 5 s interval, with robots.txt and the stop-on-403 rule still in force. Result: 1,292
  detail pages and 1,230 script pages fetched, 0 fetch errors, 0 blocked requests. **Attach the approval
  document to this record** (`permit --level bulk --evidence "…path to document…" --confirm`) so the evidence
  trail is complete.
