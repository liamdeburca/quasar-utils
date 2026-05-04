from logging import LogRecord

def quasar_filter(record: LogRecord) -> bool:
    return record.name.startswith('quasar')