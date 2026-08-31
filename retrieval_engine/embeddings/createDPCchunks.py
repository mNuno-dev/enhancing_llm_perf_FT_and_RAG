#!/usr/bin/env python3
import json
import os
import sys

#
# Usage: script.py input_dir output_dir
#
#  input_dir contains plain text files formatted as follows:
#     - first non-empty line is the page title
#     - a non-empty line preceeded by two empty lines and followed by an empty line is a section title
#     - a non-empty line is a paragraph


class Page:
    def __init__(self, title, sections):
        self.title = title
        self.sections = sections

    def as_dict(self):
        return {
            "title": self.title,
            "sections": [section.as_dict() for section in self.sections],
            "chunks": self.get_chunks()
        }

    def get_chunks(self):
        chunks = [self.title]
        for section in self.sections:
            if section.title:
                for paragraph in section.paragraphs:
                    chunks.append("\n".join([self.title, section.title, paragraph]))
            else:
                for paragraph in section.paragraphs:
                    chunks.append("\n".join([self.title, paragraph]))
        return chunks


class Section:
    def __init__(self, title, paragraphs):
        self.title = title
        self.paragraphs = paragraphs

    def as_dict(self):
        return {"title": self.title, "paragraphs": self.paragraphs}


def plain_text_to_page(lines):
    page_title = None
    section_title = None
    paragraphs = []
    sections = []
    num_preceeding_blanks = 0
    for line in lines:
        line = line.strip()
        if not line:
            num_preceeding_blanks += 1
            continue

        if page_title is None:
            page_title = line
        elif num_preceeding_blanks >= 2:
            if section_title is not None or paragraphs:
                sections.append(Section(section_title, paragraphs))
            section_title = line
            paragraphs = []
        else:
            paragraphs.append(line)
        num_preceeding_blanks = 0
    if section_title or paragraphs:
        sections.append(Section(section_title, paragraphs))
    return Page(page_title, sections)


def process_file(input_file_path, output_file_path):
    with open(input_file_path, "rt") as lines:
        page = plain_text_to_page(lines)

    with open(output_file_path, "wt") as f:
        json.dump(page.as_dict(), f, indent=4)



def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, fname)
        output_file_path = os.path.join(output_dir, fname.replace(".txt", ".json"))
        process_file(input_file_path, output_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py input_dir output_dir")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

