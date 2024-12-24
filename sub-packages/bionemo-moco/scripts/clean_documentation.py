import re


with open("documentation.md", "r") as file:
    lines = file.readlines()

# Delete lines that start with "  * " and "    * "
lines = [line for line in lines if not line.startswith("  * ") and not line.startswith("    * ")]

# Join the lines back into a string
markdown = "".join(lines)

# Replace dots with no space in anchor ids
markdown = re.sub(r'<a id="([a-zA-Z0-9_\.]+)">', lambda match: f'<a id="{match.group(1).replace(".", "")}">', markdown)

# Replace dots with no space in links
markdown = re.sub(
    r"\[([^\]]+)\]\(#([a-zA-Z0-9_\.]+)\)",
    lambda match: f'[{match.group(1)}](#{match.group(2).replace(".", "")})',
    markdown,
)

with open("documentation.md", "w") as file:
    file.write(markdown)
