import base64


class Base64Expert():
    """
    Encodes and decodes strings using Base64 encoding.
    """
    def encode(self, s):
        return base64.b64encode(s.encode()).decode()

    def decode(self, s):
        decoded = base64.b64decode(s)
        return ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in decoded.decode('latin-1'))

shift = 3
class CaesarExpert():
    """
    Encodes and decodes strings using Caesar cipher with a fixed shift.
    """
    def encode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') + shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') + shift) % 26)
            else:
                ans += p
        return ans

    def decode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') - shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') - shift) % 26)
            else:
                ans += p
        return ans
    
class UnicodeExpert():
    """
    Encodes and decodes strings using Unicode escape sequences.
    """
    def encode(self, s):
        return ''.join(f'\\u{ord(c):04x}' for c in s)

    def decode(self, s):
        parts = s.split('\\u')
        decoded = ''.join(chr(int(part, 16)) for part in parts if part)
        return decoded
    
class MorseExpert():
    """
    Encodes and decodes strings using Morse code.
    """
    def encode(self, s):
        morse_code = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
            'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
            'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
            'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
            'Y': '-.--', 'Z': '--..', 
            '0': '-----', '1': '.----', '2': '..---', 
            '3': '...--', '4': '....-', 
            '5': '.....', '6': '-....', 
            '7': '--...', '8': "---..", 
            '9': "----.", 
            ', ': '--..--', '. ': '.-.-.-', '? ': '..--..',
            '/ ': '-..-.', '- ': '-....-', '(' : '-.--.', ') ': '-.--.-'
        }
        return ' '.join(morse_code.get(c.upper(), '') for c in s)
    
    def decode(self, s):
        morse_code = {v: k for k, v in {
            '.-':'A', '-...':'B', '-.-.':'C', '-..':'D', '.':'E',
            '..-.':'F', '--.':'G', '....':'H', '..':'I', '.---':'J',
            '-.-':'K', '.-..':'L', '--':'M', '-.':'N', "---":'O',
            '.--.':'P', '--.-':'Q', '.-.':'R',  "...":'S','-':'T',
            '..-':'U','...-':'V','.--':'W','-..-':'X','-.--':'Y','--..':'Z',
            "-----":'0','.----':'1','..---':'2','...--':'3','....-':'4',
            ".....":'5','-....':'6','--...':'7','---..':'8','----.':'9',
            '--..--':',','.':'-.-.-','..--..':'?','-..-.':'/','-....-':'-',
            '-.--.':'(', '-.--.-':')'
        }.items()}
        decoded = []
        for word in s.split(' '):
            if word in morse_code:
                decoded.append(morse_code[word])
            else:
                decoded.append('')
        return ''.join(decoded)