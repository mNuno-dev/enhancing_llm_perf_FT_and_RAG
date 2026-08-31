#! /bin/bash

journalctl -k --since "1 hour ago" --output short-iso $@ |
grep -Ei 'oom|killed process|segfault'

