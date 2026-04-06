#!/usr/bin/env python3
"""
Compatibility wrapper for the VisText V3 converter.

The canonical implementation now lives in `json_to_ttl_converter_v3.py`.
This file is kept so older references still import and execute successfully.
"""

from json_to_ttl_converter_v3 import (
    DEFAULT_OUTPUT_DIR,
    ConversionResult,
    JSONToTTLConverterV3,
    JSONToTTLConverterV3 as JSONToTTLConverterScenegraph,
    main,
    parse_scenegraph,
)


if __name__ == "__main__":
    raise SystemExit(main())
