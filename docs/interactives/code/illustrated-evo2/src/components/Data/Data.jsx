import * as d3 from 'd3';
import './Data.css';
import { DataBarplot } from '../Barplot/DataBarplot';
import SideNote from '../UI/SideNote/SideNote';
import Accordion from '../UI/Accordion/Accordion';
import VariableWidthBarplot from '../Barplot/VariableWidthBarplot';

const segments = [
  { length: 10, color: '#7285b7', type: 'selfish element' },
  { length: 15, color: '#283250', type: 'selfish element' },
  { length: 10, color: '#8da4d4', type: 'selfish element' },
  { length: 10, color: '#bf7f2a', type: 'essential gene' },
  { length: 10, color: '#8da4d4', type: 'selfish element' },
  { length: 80, color: '#d0dbe8', type: 'repeat expansion' },
  { length: 10, color: '#8da4d4', type: 'selfish element' },
  { length: 15, color: '#283250', type: 'selfish element' },
  { length: 10, color: '#7285b7', type: 'selfish element' },
];

// Generate a random ATCG sequence matching the total length
const total = d3.sum(segments, d => d.length);
const bases = 'ATCG';
const sequence = d3
  .range(total)
  .map(() => bases[Math.floor(Math.random() * 4)])
  .join('');

const Data = () => {
  return (
    <>
      <h1 className="body-header">Data &amp; Training</h1>
      <p className="body-text">
        As part of the Evo 2 release,{' '}
        <a href="https://huggingface.co/datasets/arcinstitute/opengenome2">
          OpenGenome2
        </a>
        , a dataset containing non-redundant nucleotide sequence data with over
        8.8 trillion nucleotides from bacteria, archaea, eukarya, and
        bacteriophage, was also released and open-sourced.
      </p>
      <div className="body-text">
        <Accordion>
          <p className="body-text">
            <b>Bacteria</b>: Bacteria are single‑celled, prokaryotic
            microorganisms that lack a nucleus and other membrane‑bound
            organelles; they inhabit virtually every environment on Earth and
            play essential roles in nutrient cycling, symbiosis, and disease.
          </p>
          <p className="body-text">
            <b>Archaea</b>: Archaea are a domain of single‑celled, prokaryotic
            microorganisms distinct from bacteria, often thriving in extreme
            environments (e.g., hot springs, salt lakes) and characterized by
            unique membrane lipids and metabolic pathways.
          </p>
          <p className="body-text">
            <b>Eukarya</b>: Eukarya is the domain of organisms whose cells
            contain a true nucleus and membrane‑bound organelles; it includes
            animals, plants, fungi, and various protists.
          </p>
          <p className="body-text">
            <b>Bacteriophage</b>: A bacteriophage is a type of virus that
            specifically infects and replicates within bacterial cells,
            typically comprising a protein capsid enclosing DNA or RNA and using
            either lytic or lysogenic life cycles.
          </p>
          <p className="body-text">
            <b>Non‑redundant sequence data</b>: Non‑redundant sequence data is a
            curated collection of nucleotide sequences from which duplicate or
            highly similar entries have been removed, ensuring each sequence is
            unique and reducing computational and storage overhead.
          </p>
        </Accordion>
      </div>
      <p className="body-text">
        Three model variants were trained on OpenGenome2: Evo 2 1B, Evo 2 7B,
        and Evo 2 40B. Each model was optimized for next-token prediction on a
        byte-tokenized, sequence-packed version of OpenGenome2.
      </p>
      <p className="body-text">
        OpenGenome2 was itself assembled from multiple sources, including the
        original{' '}
        <a
          href="https://huggingface.co/datasets/LongSafari/open-genome"
          target="_blank"
          rel="noopener noreferrer"
        >
          OpenGenome dataset
        </a>
        , the{' '}
        <a
          href="https://www.ncbi.nlm.nih.gov/refseq/"
          target="_blank"
          rel="noopener noreferrer"
        >
          NCBI RefSeq
        </a>{' '}
        database, and the{' '}
        <a
          href="https://www.ncbi.nlm.nih.gov/genbank/"
          target="_blank"
          rel="noopener noreferrer"
        >
          GenBank
        </a>{' '}
        database. A sample of some of this data is shown below:
      </p>
      <DataBarplot />
      <p className="body-text">
        To effectively model both short-range functional features and long-range
        genomic relationships, the training process was divided into two
        distinct phases, pretraining and midtraining. Each phase applied its own
        preprocessing, augmentation, and tokenization strategy:
        <br />
        <br />
        <ol style={{ marginLeft: '2rem' }}>
          <li>
            <b>Pretraining phase:</b> The model is trained with a context-length
            of 8192 tokens.
            <SideNote>
              Evo 2 40B's pretraining stage is further split into two stages,
              first training at 1024 context for 6.6T tokens before extending to
              8192 context for 1.1T tokens. Evo 2 1B base was trained at 8192
              context length for 1T tokens.
            </SideNote>
            Data in this stage was augmented to focus on more conserved,
            information-dense regions around genes, allowing the model to focus
            primarily on functional elements (e.g. gene bodies, promoters, and
            enhancers).
          </li>
          <li>
            <b>Midtraining phase:</b> The context was extended up to 1M tokens
            with a greater proportion of entire genomes in the data mix. To
            utilize the larger context window, training sequences were
            lengthened so the model could observe and learn long-range
            dependencies across entire genomic regions during training. This was
            achieved by using a combination of positional interpolation—where
            the positional indices of tokens are downscaled—and increasing the
            base frequency of the RoPE embedding.
          </li>
        </ol>
        <br />
        Intuitively, the pretraining phase enables the model to learn
        fundamental biological structures and develop better representations of
        functional elements, while during midtraining the model learns how to
        compose them in long sequences and broadens its understanding from
        eukaryotic genes to entire genomic sequences found in the genome data
        bank.
      </p>

      <div className="chart-container-data">
        <VariableWidthBarplot />
      </div>
      <p className="body-text">
        Observe that training on NCBI genome data occurs exclusively during the
        midtraining phase, where context is extended to support long-range
        dependencies. By dividing training into pretraining and midtraining
        phases, Evo 2 is able to learn both fine-grained, single-nucleotide
        features and the broad genomic diversity needed to generalize across
        everything from small mitochondrial genomes to large, complex eukaryotic
        chromosomes.
      </p>

      <p className="body-text">
        Proxy ablation evaluations on downstream tasks were used to finalize the
        choice of data composition and augmentation.
      </p>
    </>
  );
};

export default Data;
