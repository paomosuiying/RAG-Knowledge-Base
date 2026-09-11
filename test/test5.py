from processor.import_processor import config
from processor.import_processor.config import get_config

langth = get_config().max_content_length
print(langth)