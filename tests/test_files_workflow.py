"""Offline checks for files_workflow search. Run: python tests/test_files_workflow.py"""
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # core/, where workflows/ and guards.py live
from workflows import files_workflow as fw


def touch(root, rel, age_days=0):
    p = Path(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))
    return str(p)


def find(root, terms, extensions=(), index=()):
    """Run the real search node; the index returns `index`, the scan covers only root."""
    fw.search_index, fw.drives = (lambda groups, ext: list(index)), (lambda: [root])
    return fw.search({"search": {"terms": terms, "extensions": list(extensions)}})


def listed(r):
    return [line.split("  (modified")[0].strip() for line in r["output"].splitlines()[1:]]


def test_all():
    with tempfile.TemporaryDirectory() as root:
        old = touch(root, "Documents/Shakthi_Resume_2023.pdf", age_days=400)
        new = touch(root, "Downloads/ShakthiResume.pdf", age_days=1)
        cv = touch(root, "Desktop/jobs/cv final.docx", age_days=5)
        touch(root, "Desktop/app/node_modules/resume-parser/resume.js")  # dependency folder: skipped
        touch(root, "Documents/.cache/resume.pdf")                        # hidden folder: skipped
        touch(root, "AppData/Local/resume.pdf")                           # app data: skipped
        folder = str(Path(root, "Projects/resume-site")); os.makedirs(folder)
        t = time.time() - 10 * 86400; os.utime(folder, (t, t))

        # alternatives in one group: resume OR cv; folders count; newest first; skipped folders never show up
        r = find(root, [["resume", "cv"]])
        assert not r["failed"] and r["output"].startswith("Found 4 matching resume/cv:"), r
        assert listed(r) == [new, cv, folder, old], listed(r)
        assert "node_modules" not in r["output"] and ".cache" not in r["output"]
        assert str(Path(root, "AppData")) not in r["output"]  # (root itself sits under the real AppData\Temp)

        # extension filter, given with or without the dot, any case; it applies to the name, not the folder
        r = find(root, [["resume"]], ["PDF"])
        assert r["output"].startswith("Found 2 matching resume (.pdf):") and listed(r) == [new, old], r

        # several groups: Found only when the name has every one of them
        r = find(root, [["shakthi"], ["resume", "cv"], ["2023"]])
        assert r["output"].startswith("Found 1 matching shakthi + resume/cv + 2023:") and listed(r) == [old], r
        # no exact match: says so plainly, then lists the closest names, most groups matched first
        r = find(root, [["acme"], ["resume"], ["2023"]])
        assert r["output"].startswith("No file matching all of acme + resume + 2023 on this computer. Closest"), r
        assert listed(r)[0] == old, listed(r)

        # terms can sit in the parent folders: camera files have no useful names
        photo = touch(root, "Pictures/Goa Trip 2025/IMG_0042.jpg")
        touch(root, "Pictures/Goa Trip 2025/notes.txt")
        r = find(root, [["goa"], ["trip"]], [".jpg", ".png"])
        assert r["output"].startswith("Found 1 matching goa + trip (.jpg .png):") and listed(r) == [photo], r

        # exact file name, kept whole
        report = touch(root, "Work/report_final.docx")
        touch(root, "Work/report_draft.docx")
        assert listed(find(root, [["report_final"]], [".docx"])) == [report]

        # an exact hit from the index is used as is (no scan); no exact hit there means the drives get scanned
        assert listed(find(root, [["resume"]], index=[old])) == [old]
        r = find(root, [["report_final"]], index=[])
        assert listed(r) == [report], r

        # nothing found says so, and what was searched; it is an answer, not a failure
        r = find(root, [["tax"], ["return"]])
        assert not r["failed"] and r["output"].startswith("No file matching tax + return on this computer."), r
        assert root in r["output"], r

        # nothing to search for is refused instead of listing the whole disk
        assert find(root, [[" "]])["output"].startswith("refused: no file name")

        # more than MAX_RESULTS hits: the count is honest, the list is capped
        for i in range(fw.MAX_RESULTS + 3):
            touch(root, f"Bulk/quarterly_{i}.txt")
        r = find(root, [["quarterly"]])
        assert f"Found {fw.MAX_RESULTS + 3} matching quarterly, showing the best {fw.MAX_RESULTS}:" in r["output"], r
        assert len(listed(r)) == fw.MAX_RESULTS


def test_index():
    """The real Windows index: a quote can't break the SQL, and results are exact matches only.
    Returns [] where there is no index, which the search treats as 'scan instead'."""
    real = importlib.reload(fw)  # undo find()'s patches
    assert isinstance(real.search_index([["o'brien'; DROP"]], [".pdf"]), list)
    for p in real.search_index([["py"], ["test"]], [".py"]):
        assert real.score(p, [["py"], ["test"]]) == 2, p


if __name__ == "__main__":
    test_all()
    test_index()
    print("all checks passed")
