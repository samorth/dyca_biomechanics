import os
import yaml
import logging.config
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

current_subject = {'id': 'UNKNOWN'}

class SubjectFilter(logging.Filter):
    def filter(self, record):
        record.subject_id = current_subject['id']
        return True
    

class RawConfig(BaseModel):
    columns_to_drop: List[str]
    marker_loc: Dict[str, List[List[int]]]
    default_preprocessing: Dict[str, bool]
    logging: Dict[str, Any]

    @staticmethod
    def _find_config(filename: str = "config/default.yaml"):
        base = Path(__file__).resolve().parent
        for parent in [base, *base.parents]:
            candidate = parent / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Konfigurationsdatei '{filename}' nicht gefunden ab {base} aufwärts")

    @classmethod
    def load(cls, path: str = None):
        if path:
            cfg_path = Path(path)
        elif os.getenv("CONFIG_FILE"):
            cfg_path = Path(os.getenv("CONFIG_FILE"))
        else:
            cfg_path = cls._find_config()

        if not cfg_path.is_file():
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

settings = RawConfig.load()


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"
PROJECT_ROOT = CONFIG_PATH.parent.parent

old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.subject_id = current_subject.get('id', 'UNKNOWN')
    return record
logging.setLogRecordFactory(record_factory)

def setup_logging():
    cfg = settings.logging
    
    for h in cfg.get("handlers", {}).values():
        fn = h.get("filename")
        if fn:
            p = Path(fn)
            if not p.is_absolute():
                abs_p = (PROJECT_ROOT / p).resolve()
                h["filename"] = str(abs_p)
                p = abs_p
            p.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(cfg)
    logging.getLogger().addFilter(SubjectFilter())