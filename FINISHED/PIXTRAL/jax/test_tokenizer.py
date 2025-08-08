from display_common import print_color, color_red, color_green


import json
from mistral_common.tokens.tokenizers.tekken import Tekkenizer 
tok = Tekkenizer.from_file("./pixtral/tekken.json")

def tek_encode(string, add_special=False):
  return tok.encode(string, bos=add_special, eos=add_special)

def tek_decode(string):
  return tok.decode(string)



from forward_common import encode, decode
                        


strings = [
    "Hello, how are you",
    "Whats   good🫠🫠🫠",
    "<unk> lmao",
    "🫠🫠🫠🫠你好世界 café mañana",
    """   leading space
    trailing space   
    line1\nline2
    tabs\tare\there
    """,
    """Null:\x00
    Bell:\x07
    Mix:\x00\x07Hello
    """,
    "supercalifragilisticexpialidocious",
    "\xff\xfe\xfd",
    "<s>🫠 Hello 世界</s>",
    "!!!",
    "abc!!!xyz",
    "123,456.789",
    "Describe the image.",
]



print("Encoding:")
print("Original => tekken || mine")
all_passed = True
for string in strings:
    tek_encoded_string = tek_encode(string)
    tek_decoded_string = tek_decode(tek_encoded_string)
    print(f"{string} =(tek )> {tek_encoded_string} => {tek_decoded_string}")
    my_encoded_string = encode(string)
    my_decoded_string = decode(my_encoded_string)
    print(f"{string} =(mine)> {my_encoded_string} => {my_decoded_string}")
    passed = tek_decoded_string == my_decoded_string
    passed = passed and (tek_encoded_string == my_encoded_string)
    all_passed = passed and all_passed
    if passed:
        print_color(">>>", passed, "<<<", color=color_green)
    else:
        print_color("!!!", passed, "!!!", color=color_red)
        print("correct tokenization: ", [tek_decode([tok_id]) for tok_id in tek_encoded_string])
        print("my tokenization: ", [tek_decode([tok_id]) for tok_id in my_encoded_string])
    print()


if all_passed:
    print_color("ALL TESTS PASSED", color=color_green)
else:
    print_color("TESTS FAILED", color=color_red)




print("Fuzzer test")
import random
import string

# I had ai make this function
def random_unicode_string(max_len=50):
    # Buckets of different codepoint ranges
    ascii_chars = [chr(i) for i in range(0x20, 0x7F)]
    whitespace_chars = list(" \t\n\r")
    emoji_chars = [chr(i) for i in range(0x1F300, 0x1F600)] + \
                  [chr(i) for i in range(0x1F600, 0x1F650)]
    cjk_chars = [chr(i) for i in range(0x4E00, 0x9FFF)]
    combining_marks = [chr(i) for i in range(0x0300, 0x036F)]
    specials = ["\u200B", "\u200D", "\uFEFF"]  # zero-width space/joiner, BOM

    pool = ascii_chars + whitespace_chars + emoji_chars + cjk_chars + combining_marks + specials

    length = random.randint(1, max_len)
    s = "".join(random.choice(pool) for _ in range(length))
    return s

# Example: generate 5 fuzz strings
all_fuzzer_passed = True
test_count = 1000
for i in range(test_count):
    string = random_unicode_string(max_len=500)
    tek_encoded_string = tek_encode(string)
    tek_decoded_string = tek_decode(tek_encoded_string)
    #print(f"{string} =(tek )> {tek_encoded_string} => {tek_decoded_string}")
    my_encoded_string = encode(string)
    my_decoded_string = decode(my_encoded_string)
    #print(f"{string} =(mine)> {my_encoded_string} => {my_decoded_string}")
    passed = tek_decoded_string == my_decoded_string
    passed = passed and (tek_encoded_string == my_encoded_string)
    all_fuzzer_passed = passed and all_passed
    print(f"test {i} ", end="", flush=True)
    if passed:
        print_color(">>>", passed, "<<<", color=color_green)
    else:
        print_color("!!!", passed, "!!!", color=color_red)
        #print("correct tokenization: ", [tek_decode([tok_id]) for tok_id in tek_encoded_string])
        #print("my tokenization: ", [tek_decode([tok_id]) for tok_id in my_encoded_string])

if all_passed:
    print_color("ALL FUZZER TESTS PASSED", color=color_green)
else:
    print_color("FUZZER TESTS FAILED", color=color_red)

