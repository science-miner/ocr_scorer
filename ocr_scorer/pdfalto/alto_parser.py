import os
import xml
from xml.sax import make_parser

ALTO_NS = 'http://www.loc.gov/standards/alto/ns-v3#'
ALTO_NS_MAP = {
    'alto': ALTO_NS
}

LOGGER = logging.getLogger(__name__)

class ALTOContentHandler(xml.sax.ContentHandler):
    """
    Minimalistic parser for ALTO file, just filtering text content
    """

    text = ""

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == 'SP':
            self.text += " "
        elif name == 'String':
            self.text += attrs.get('CONTENT') or ''
        elif name == 'TextLine':
            self.text += "\n"
        elif name == 'TextBlock':
            self.text += "\n"
        elif name == 'Page':
            self.text += "\n"
        
    def get_text(self):
        return self.text.strip()

def filter_text(alto_file):
    parser = make_parser()
    handler = ALTOContentHandler()
    parser.setContentHandler(handler)
    parser.parse(alto_file)
    return handler.get_text()
