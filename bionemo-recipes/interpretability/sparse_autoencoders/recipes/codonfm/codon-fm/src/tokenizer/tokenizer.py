import re
from itertools import product
from pathlib import Path

from transformers import PreTrainedTokenizer

from .mappings import CODON_TABLE, AA_TABLE

class Tokenizer(PreTrainedTokenizer):
    """
    Tokenizer: tokenize 3-mer codons into tokens
    The input sequences are expected to be raw sequences of coding DNA/RNA sequences.
    """

    SUPPORTED_TYPES = ["dna", "rna"]
    
    @staticmethod
    def get_all_codons(seq_type="dna"):
        """
        Get all possible codons.
        """
        seq_type = seq_type.lower()
        assert (
            seq_type in Tokenizer.SUPPORTED_TYPES
        ), f"seq_type should be either 'dna' or 'rna'. Got {seq_type}!"

        if seq_type == "dna":
            return ["".join(codon) for codon in product("ACGT", repeat=3)]
        else:
            return ["".join(codon) for codon in product("ACGU", repeat=3)]

    def __init__(
        self,
        cls_token="<CLS>",
        sep_token="<SEP>",
        unk_token="<UNK>",
        pad_token="<PAD>",
        mask_token="<MASK>",
        seq_type="dna",
        organism_tokens_file=None,
        **kwargs,
    ):
        self.codons = self.get_all_codons(seq_type=seq_type)
        self.seq_type = seq_type
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        self.special_tokens = (
            [cls_token, sep_token, unk_token, pad_token, mask_token]
        )

        self.encoder = {k: i for i, k in enumerate(self.special_tokens + self.codons)}
        self.decoder = {i: k for k, i in self.encoder.items()}
        self.compiled_regex = re.compile(
            "|".join(self.codons + self.special_tokens + [r"\S"])
        )
        
        # - NOTE: below should happen after the above get added to the dictionary
        # Read organism tokens from file if provided
        if organism_tokens_file is not None:
            organism_tokens_path = Path(organism_tokens_file)
            if organism_tokens_path.exists():
                with open(organism_tokens_file, "r") as f:
                    organism_tokens = [line.strip() for line in f if line.strip()]
                self.set_organism_tokens(organism_tokens)
            else:
                print(f"Warning: Organism tokens file not found: {organism_tokens_file}")
                self.organism_tokens = []
        else:
            self.organism_tokens = []
            
        super().__init__(
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.aa_to_codon = AA_TABLE[seq_type]
        self.codon_to_aa = CODON_TABLE[seq_type]
        self.encoder_aa = {
            k: i for i, k in enumerate(self.special_tokens + self.amino_acids)
        }
        self.compiled_regex_aa = re.compile(
            "|".join(self.amino_acids + self.special_tokens + [r"\S"])
        )

        self.token_type_mode = kwargs.get("token_type_mode", "regular")
        self.build_token_type_encoder()
        
    def set_organism_tokens(self, organism_tokens):
        """
        Add organism tokens to the tokenizer.
        """
        vocab_size = len(self.encoder)
        for i, token in enumerate(organism_tokens):
            self.encoder[token] = vocab_size + i
            self.decoder[vocab_size + i] = token

        self.organism_tokens = organism_tokens
        self.compiled_regex = re.compile(
            "|".join(self.codons + self.special_tokens + organism_tokens + [r"\S"])
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def build_token_type_encoder(self):
        if self.token_type_mode == "aa":
            # build a token type encoder for amino acids with codon ids as keys and amino acid ids as values
            # CLS, SEP, UNK, MASK, PAD tokens are assigned to the same token type as zero
            token_type_encoder = {}
            for token, token_id in self.encoder.items():
                if token in self.special_tokens:
                    token_type_encoder[token_id] = 0
                elif token in self.codons:
                    aa = self.codon_to_aa[token]
                    token_type_encoder[token_id] = (
                        list(self.amino_acids + ["*"]).index(aa) + 1
                    )
                else:
                    token_type_encoder[token_id] = len(self.amino_acids) + 2
        elif self.token_type_mode == "regular":
            # build a token type encoder for regular tokens
            token_type_encoder = {token_id: 0 for token_id in self.encoder.values()}
        elif self.token_type_mode == "regular_special":
            # build a token type encoder for regular tokens with special tokens having a different but same token type
            token_type_encoder = {
                token_id: 0 if token in self.special_tokens else 1
                for token, token_id in self.encoder.items()
            }
        else:
            raise ValueError(f"Unknown token type mode: {self.token_type_mode}")

        self.token_type_encoder = token_type_encoder

    @property
    def token_type_vocab_size(self):
        return len(set(self.token_type_encoder.values())) + 1

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """
        Tokenize a string.
        """
        text = text.upper()
        tokens = self.compiled_regex.findall(text)
        return tokens

    def _convert_token_to_id(self, token):
        """
        Converts a token (str) in an id using the vocab.
        """
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab.
        """
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        """
        return "".join(tokens)

    def encode_aa(self, text):
        """
        Encode a DNA/RNA string using the amino acid vocab.
        """
        tokens = self._tokenize(text)
        return [
            self.encoder_aa.get(token, self.encoder_aa[self.unk_token])
            for token in tokens
        ]
    
    def encode(self, text: str, **kwargs):
        """
        Encode a DNA/RNA string using the DNA vocab.
        """
        tokens = self._tokenize(text)
        return [self.encoder.get(token, self.encoder[self.unk_token]) for token in tokens]

    def get_aa_vocab_size(self):
        return len(self.encoder_aa)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        token_ids_0 = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return token_ids_0

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens: bool = False
    ):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        special_ids = [
            self.eos_token_id,
            self.pad_token_id,
            self.mask_token_id,
            self.sep_token_id,
            self.cls_token_id,
        ]

        if already_has_special_tokens:
            special_tokens_mask = [
                1 if idx in special_ids else 0 for idx in token_ids_0
            ]
        else:
            special_tokens_mask = (
                [1] + [1 if idx in special_ids else 0 for idx in token_ids_0] + [1]
            )

        return special_tokens_mask

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """

        unk_type_id = len(set(self.token_type_encoder.values()))

        token_type_ids = [
            self.token_type_encoder.get(token_id, unk_type_id)
            for token_id in token_ids_0
        ]

        return token_type_ids