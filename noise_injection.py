import random
import string

QWERTY_MAP = {
    'q': ['w', 'a', 's'],
    'w': ['q', 'e', 'a', 's', 'd'],
    'e': ['w', 'r', 's', 'd', 'f'],
    'r': ['e', 't', 'd', 'f', 'g'],
    't': ['r', 'y', 'f', 'g', 'h'],
    'y': ['t', 'u', 'g', 'h', 'j'],
    'u': ['y', 'i', 'h', 'j', 'k'],
    'i': ['u', 'o', 'j', 'k', 'l'],
    'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z', 'x'],
    's': ['a', 'w', 'e', 'd', 'z', 'x', 'c'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c', 'v'],
    'f': ['d', 'r', 't', 'g', 'c', 'v', 'b'],
    'g': ['f', 't', 'y', 'h', 'v', 'b', 'n'],
    'h': ['g', 'y', 'u', 'j', 'b', 'n', 'm'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k'],
}

def inject_qwerty_noise(text, noise_level):
    """
    Inject QWERTY keyboard proximity noise into text.
    Following Gupta et al. 2020: randomly replace k% of characters
    with adjacent QWERTY keys.
    
    Args:
        text: input string
        noise_level: percentage of characters to replace (e.g., 5 for 5%)
    
    Returns:
        noisy text string
    """
    if noise_level == 0:
        return text
    
    chars = list(text)
    # Get indices of alphabetic characters only
    alpha_indices = [i for i, c in enumerate(chars) if c.lower() in QWERTY_MAP]
    
    if not alpha_indices:
        return text
    
    # Number of characters to replace
    n_replace = max(1, int(len(chars) * noise_level / 100))
    n_replace = min(n_replace, len(alpha_indices))
    
    # Randomly select indices to replace
    indices_to_replace = random.sample(alpha_indices, n_replace)
    
    for idx in indices_to_replace:
        original_char = chars[idx]
        lower_char = original_char.lower()
        
        if lower_char in QWERTY_MAP:
            replacement = random.choice(QWERTY_MAP[lower_char])
            # Preserve original case
            if original_char.isupper():
                replacement = replacement.upper()
            chars[idx] = replacement
    
    return ''.join(chars)


def generate_noisy_dataset(texts, noise_level, seed=42):
    """
    Generate a noisy version of a list of texts at a given noise level.
    
    Args:
        texts: list of input strings
        noise_level: percentage noise (0, 5, 10, 15, 20, 25)
        seed: random seed for reproducibility
    
    Returns:
        list of noisy strings
    """
    random.seed(seed)
    return [inject_qwerty_noise(text, noise_level) for text in texts]


def generate_all_variants(texts, noise_levels=[0, 5, 10, 15, 20, 25], seed=42):
    """
    Generate all noisy variants D0, D5, D10, D15, D20, D25.
    
    Args:
        texts: list of input strings
        noise_levels: list of noise percentages to generate
        seed: random seed for reproducibility
    
    Returns:
        dict mapping noise_level -> list of noisy texts
    """
    return {
        level: generate_noisy_dataset(texts, level, seed=seed)
        for level in noise_levels
    }


if __name__ == "__main__":
    # Quick test on 5 samples
    test_texts = [
        "that loves its characters and communicates something rather beautiful about human nature.",
        "poor ben bratt could not find stardom if mapquest emailed him directions.",
        "this movie was absolutely terrible and a complete waste of time.",
        "the acting was superb and the storyline kept me engaged throughout.",
        "a masterpiece of modern cinema with breathtaking visuals and performances."
    ]
    
    print("=== QWERTY Noise Injection Test ===\n")
    print(f"{'Noise Level':<15} {'Sample Text'}")
    print("-" * 80)
    
    for level in [0, 5, 10, 15, 20, 25]:
        random.seed(42)
        noisy = inject_qwerty_noise(test_texts[0], level)
        print(f"D{level:<14} {noisy}")
    
    print("\n=== All 5 samples at 10% noise ===\n")
    noisy_variants = generate_all_variants(test_texts)
    for i, text in enumerate(noisy_variants[10]):
        print(f"[{i+1}] {text}")
    
    print("\nPipeline working correctly.")
