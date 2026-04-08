import os
import re
import json
import datetime
from typing import Optional

from config import BASE_GEN_DIR


def parse_generated_content(response_text: str) -> list[dict[str, str]]:
    """
    Parses the raw LLM response into structured dictionaries.
    Robustly handles splitting by '# id:' and filters out artifacts like '---'.
    """
    examples = []

    id_pattern = re.compile(r"^# id:\s*(?P<id>[\w-]+)", re.MULTILINE)
    matches = list(id_pattern.finditer(response_text))

    for i, match in enumerate(matches):
        file_id = match.group("id")
        start_index = match.start()

        if i + 1 < len(matches):
            end_index = matches[i + 1].start()
        else:
            end_index = len(response_text)

        chunk = response_text[start_index:end_index].strip()

        lines = chunk.splitlines()
        metadata_lines = []
        code_lines = []

        capture_code = False

        for line in lines:
            if "---" in line and len(line.strip()) < 5:
                continue

            stripped = line.strip()

            if stripped.startswith("# id:"):
                continue

            if not capture_code:
                if stripped.startswith("#"):
                    metadata_lines.append(line)
                elif stripped == "" or stripped.startswith("```"):
                    continue
                else:
                    capture_code = True
                    code_lines.append(line)
            else:
                if stripped.startswith("```"):
                    continue
                code_lines.append(line)

        full_code = "\n".join(code_lines).strip()
        full_metadata = "\n".join(metadata_lines).strip()

        if file_id and full_code:
            examples.append(
                {
                    "id": file_id,
                    "metadata": full_metadata,
                    "code": full_code,
                    "full_content": f"# id: {file_id}\n{full_metadata}\n\n{full_code}",
                }
            )

    return examples


def save_output(
    examples: list[dict[str, str]], raw_response: str, model_name: str
) -> Optional[str]:
    """
    Saves the parsed examples to JSON and individual .py files.
    Returns the base_path of the created directory.
    """
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    base_path = os.path.join(BASE_GEN_DIR, folder_name)
    source_files_path = os.path.join(base_path, "source_files")

    os.makedirs(source_files_path, exist_ok=True)
    print(f"\n[INFO] Created output directory: {base_path}")

    for ex in examples:
        filename = f"{ex['id']}.py"
        file_path = os.path.join(source_files_path, filename)

        # Write only the code without metadata comments
        # to avoid biasing the evaluation LLM.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(ex["code"])
        print(f"  -> Saved {filename}")

    json_path = os.path.join(base_path, "examples.json")

    output_data = {
        "timestamp": now.isoformat(),
        "model_used": model_name,
        "raw_response": raw_response,
        "examples": examples,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"[INFO] Saved master JSON to: {json_path}")
    print(f"[INFO] Successfully saved {len(examples)} examples.")

    return base_path

