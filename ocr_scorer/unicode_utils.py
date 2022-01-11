import regex as re

'''
Normalization of unicode string, particularly useful when text is comming from PDF. This is a direct port of this:
https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/utilities/UnicodeUtil.java
See the reference Java class for more detailed comments on the involved unicode points. 

The normalization replaces typical glyph variants of common punctuations/separators by a canonical one, e.g. all 
spaces by the standard white space, all bullet variations by '\u2022' etc. This will help any further text processing
without any visible impact. 
'''

# python re or regex do not seem to support unicode character class/property (or I did not find proper documentation on this)
# and all my tries failed. So I enumerate the official code points of unicode properties. 
dash_chars = r"[" \
    + "\u002D" \
    + "\u058A" \
    + "\u05BE" \
    + "\u1806" \
    + "\u2010" \
    + "\u2011" \
    + "\u2012" \
    + "\u2013" \
    + "\u2014" \
    + "\u2015" \
    + "\u2E17" \
    + "\u2E1A" \
    + "\u2E3A" \
    + "\u2E3B" \
    + "\u301C" \
    + "\u3030" \
    + "\uFE58" \
    + "\uFE63" \
    + "\uFF0D" \
    + "]"
DASH_PATTERN = re.compile(dash_chars)

NORMALIZE_REGEX_PATTERN = re.compile(r"[ \n]+")

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
Normalize the space, EOL and punctuation unicode characters.

In particular all the characters which are treated as space in
C++ (http://en.cppreference.com/w/cpp/string/byte/isspace)
will be replace by the punctuation space character U+2008
so that the token can be used to generate a robust feature vector
legible as Wapiti input.

@param text to be normalized
@return normalized string
'''
def normalize_text(text):
    if text == None:
        return None

    if text == "\\":
        return text

    # in Python if the string ends with "\" or "\\", this is going to end badly with regex application here :)
    # one backslash "\" should be replaced with "\\\\"
    if text.endswith("\\\\"):
        text += "\\\\"
    elif text.endswith("\\"):
        text += "\\\\\\"

    # normalize all horizontal space separator characters 
    text = MY_WHITESPACE_PATTERN.sub(" ", text);

    # normalize all EOL - special handling of "\r\n" as one single newline
    text = text.replace("\r\n", r"\n")
    text = NEW_LINE_CHARS_PATTERN.sub("\n", text)

    # normalize dash via the unicode dash punctuation property
    text = DASH_PATTERN.sub("-", text)

    # normalize horizontal low lines
    text = HORIZONTAL_LOW_LINES_CHARS_PATTERN.sub("_", text)

    # normalize vertical lines
    text = VERTICAL_LINES_CHARS_PATTERN.sub("|", text)

    # bullet normalisation
    text = BULLET_CHARS_PATTERN.sub("•", text)

    # opening parenthesis normalisation
    text = OPEN_PARENTHESIS_PATTERN.sub("(", text)

    # closing parenthesis normalisation
    text = CLOSE_PARENTHESIS_PATTERN.sub(")", text)

    # remove all control charcaters?
    # text = re.replace(r"\p{Cntrl}", " ", text)

    return text;
    
'''
Unicode normalisation of text.
Works as the {@link org.grobid.core.utilities.UnicodeUtil#normalizeText(java.lang.String)}, 
but in addition also replace any spaces+EOL sequences by a single space
@param text to be normalized
@return normalized string
'''
def normalize_text_and_collapse_spaces(text):
    # parano sanitising
    return NORMALIZE_REGEX_PATTERN.matcher(normalizeText(text)).replaceAll(" ")


if __name__ == '__main__':
    # run some test
    string = "‑‒–—―"
    string = normalize_text(string)
    if string != "-----":
        print("error: failed to normalized dash pattern", "‑‒–—―", string)
    else:
        print("full dash text pass")

    string = "—―"
    string = normalize_text(string)
    if string != "--":
        print("error: failed to normalized dash pattern", "—―", string)
    else:
        print("double dash text pass")

    string = "‒"
    string = normalize_text(string)
    if string != "-":
        print("error: failed to normalized dash pattern", "‒", string)
    else:
        print("single dash text pass")

    string = "•‣◦⁃⁌⁍∙◘⦾⦿⏺●⚫⬤·"
    string = normalize_text(string)
    if string != "•••••••••••••••":
        print("error: failed to normalized dash pattern", "•‣◦⁃⁌⁍∙◘⦾⦿⏺●⚫⬤·", string)
    else:
        print("full bullet text pass")

    string = "//"
    string = normalize_text(string)
    if string != "//":
        print("error: failed to normalized single backslas pattern", "//", string)
    else:
        print("single backslash text pass")

    string = r"/"
    string = normalize_text(string)
    if string != r"/":
        print("error: failed to normalized raw single backslash pattern", r"/", string)
    else:
        print("raw single backslash text pass")

    string = "////"
    string = normalize_text(string)
    if string != "////":
        print("error: failed to normalized double backslash  pattern", "////", string)
    else:
        print("double backslash text pass")
