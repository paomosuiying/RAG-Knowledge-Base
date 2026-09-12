from processor.import_processor import import_config
from processor.import_processor.import_config import get_config

langth = get_config().max_content_length
print(langth)