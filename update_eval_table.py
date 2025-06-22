import json

with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Build a Markdown table
markdown = "# Model Evaluation\n\n"
markdown += "| Metric | Value |\n"
markdown += "|--------|--------|\n"
for key, value in metrics.items():
    markdown += f"| {key} | {value} |\n"

# Write to EVAL.md
with open("EVAL.md", "w") as f:
    f.write(markdown)

print("EVAL.md updated!")