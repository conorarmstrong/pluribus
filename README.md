# iPhone Backup Extractor

A Python script to extract and restore files from iTunes iPhone backups using `Manifest.db`.

## Features
- Extracts files from iTunes/Finder iPhone backups.
- Reconstructs original file paths.
- Uses SQLite to read `Manifest.db`.
- Handles missing files/errors gracefully.
- Progress bar for large backups.

## Installation
Ensure you have Python 3 installed, then install the required dependencies:

```bash
pip install tqdm
```

## Usage

### Basic Extraction
Extracts files from the iPhone backup in `/path/to/iphone/backup` to a folder called `extracted`:

```bash
python iphone_extract_backup.py "/path/to/iphone/backup"
```

### Specify Output Folder
Extracts files to `/path/to/output` instead of the default `extracted`:

```bash
python iphone_extract_backup.py "/path/to/iphone/backup" -o "/path/to/output"
```

### Disable Progress Bar
Useful for running the script in non-interactive environments:

```bash
python iphone_extract_backup.py "/path/to/iphone/backup" --no-progress
```

## Repository
GitHub: [iPhone Backup Extractor](https://github.com/conorarmstrong/iphone_extract_backup)

## Author
**Conor Armstrong**  
📧 conorarmstrong@gmail.com
