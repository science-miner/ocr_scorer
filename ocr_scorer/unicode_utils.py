import regex as re

'''
Normalization of unicode string, particularly useful when text is comming from PDF. This is a direct port of this:
https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/utilities/UnicodeUtil.java
See the reference Java class for more detailed comments on the involved unicode points. 

The normalization replaces typical glyph variants of common punctuations/separators by a canonical one, e.g. all 
spaces by the standard white space, all bullet variations by '\u2022' etc. This will help any further text processing
without any visible impact. 
'''
DASH_PATTERN = re.compile(r"\p{Pd}")
NORMALISE_REGEX_PATTERN = re.compile(r"[ \n]+")

# here are the 26 code points of the "official" stable
# p{White_Space} unicode property - not used, but here as reference/alternative
whitespace_chars = r"[" \
    + "\u0009" \
    + "\u000A" \
    + "\u000B" \
    + "\u000C" \
    + "\u000D" \
    + "\u0020" \
    + "\u0085" \
    + "\u00A0" \
    + "\u1680" \
    + "\u180E" \
    + "\u2000" \
    + "\u2001" \
    + "\u2002" \
    + "\u2003" \
    + "\u2004" \
    + "\u2005" \
    + "\u2006" \
    + "\u2007" \
    + "\u2008" \
    + "\u2009" \
    + "\u200A" \
    + "\u2028" \
    + "\u2029" \
    + "\u202F" \
    + "\u205F" \
    + "\u3000" \
    + "]"
    
# a more restrictive selection of horizontal white space characters than the
# Unicode p{White_Space} property (which includes new line and vertical spaces)
my_whitespace_chars = r"[" \
    + "\u0009" \
    + "\u0020" \
    + "\u00A0" \
    + "\u1680" \
    + "\u180E" \
    + "\u2000" \
    + "\u2001" \
    + "\u2002" \
    + "\u2003" \
    + "\u2004" \
    + "\u2005" \
    + "\u2006" \
    + "\u2007" \
    + "\u2008" \
    + "\u2009" \
    + "\u200A" \
    + "\u2028" \
    + "\u2029" \
    + "\u202F" \
    + "\u205F" \
    + "\u3000" \
    + "]"
MY_WHITESPACE_PATTERN = re.compile(my_whitespace_chars);

# all the horizontal low lines
horizontal_low_lines_chars = r"[" \
    + "\u005F" \
    + "\u203F" \
    + "\u2040" \
    + "\u2054" \
    + "\uFE4D" \
    + "\uFE4E" \
    + "\uFE4F" \
    + "\uFF3F" \
    + "\uFE33" \
    + "\uFE34" \
    + "]"
HORIZONTAL_LOW_LINES_CHARS_PATTERN = re.compile(horizontal_low_lines_chars);

# all the vertical lines
vertical_lines_chars = r"[" \
    + "\u007C" \
    + "\u01C0" \
    + "\u05C0" \
    + "\u2223" \
    + "\u2758" \
    + "]"
VERTICAL_LINES_CHARS_PATTERN = re.compile(vertical_lines_chars);

# all new lines / "vertical" white spaces
new_line_chars = r"[" \
    + "\u000C" \
    + "\u000A" \
    + "\u000D" \
    + "\u000B" \
    + "\u0085" \
    + "\u2029" \
    + "\u2028" \
    + "]"
NEW_LINE_CHARS_PATTERN = re.compile(new_line_chars);

# all bullets
bullet_chars = r"[" \
    + "\u2022" \
    + "\u2023" \
    + "\u25E6" \
    + "\u2043" \
    + "\u204C" \
    + "\u204D" \
    + "\u2219" \
    + "\u25D8" \
    + "\u29BE" \
    + "\u29BF" \
    + "\u23FA" \
    + "\u25CF" \
    + "\u26AB" \
    + "\u2B24" \
    + "\u00B7" \
    + "]"
BULLET_CHARS_PATTERN = re.compile(bullet_chars);

 # opening parenthesis
open_parenthesis = r"[" \
    + "\u0028" \
    + "\uFF08" \
    + "\u27EE" \
    + "\u2985" \
    + "\u2768" \
    + "\u276A" \
    + "\u27EC" \
    + "]"
OPEN_PARENTHESIS_PATTERN = re.compile(open_parenthesis);

# closing parenthesis
close_parenthesis = r"[" \
    + "\u0029" \
    + "\uFF09" \
    + "\u27EF" \
    + "\u2986" \
    + "\u2769" \
    + "\u276B" \
    + "\u27ED" \
    + "]"
CLOSE_PARENTHESIS_PATTERN = re.compile(close_parenthesis);

'''
Normalise the space, EOL and punctuation unicode characters.

In particular all the characters which are treated as space in
C++ (http://en.cppreference.com/w/cpp/string/byte/isspace)
will be replace by the punctuation space character U+2008
so that the token can be used to generate a robust feature vector
legible as Wapiti input.

@param text to be normalised
@return normalised string
'''
def normalise_text(text):
    if text == None:
        return None

    # normalise all horizontal space separator characters 
    text = MY_WHITESPACE_PATTERN.sub(text, " ");

    # normalise all EOL - special handling of "\r\n" as one single newline
    text = text.replace("\r\n", "\n")
    text = NEW_LINE_CHARS_PATTERN.sub(text, "\n")

    # normalize dash via the unicode dash punctuation property
    text = DASH_PATTERN.sub(text, "-")

    # normalize horizontal low lines
    text = HORIZONTAL_LOW_LINES_CHARS_PATTERN.sub(text, "_")

    # normalize vertical lines
    text = VERTICAL_LINES_CHARS_PATTERN.sub(text, "|")

    # bullet normalisation
    text = BULLET_CHARS_PATTERN.sub(text, "•")

    # opening parenthesis normalisation
    text = OPEN_PARENTHESIS_PATTERN.sub(text, "(")

    # closing parenthesis normalisation
    text = CLOSE_PARENTHESIS_PATTERN.sub(text, ")")

    # remove all control charcaters?
    # text = re.replace(r"\p{Cntrl}", text, " ")

    return text;
    
'''
Unicode normalisation of text.
Works as the {@link org.grobid.core.utilities.UnicodeUtil#normaliseText(java.lang.String)}, 
but in addition also replace any spaces+EOL sequences by a single space
@param text to be normalised
@return normalised string
'''
def normalise_text_and_collapse_spaces(text):
    # parano sanitising
    return NORMALISE_REGEX_PATTERN.matcher(normaliseText(text)).replaceAll(" ")
