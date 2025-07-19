"""Logging configuration."""

import os
import sys
import logging


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - [%(name)s], (%(levelname)s): %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join('log', 'rag_agent.log'))
        ]
    )