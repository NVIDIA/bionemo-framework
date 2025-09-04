import BeamSearch1 from './BeamSearch1';
import BeamSearch2 from './BeamSearch2';
import BeamSearch3 from './BeamSearch3';
import BeamSearch4 from './BeamSearch4';
import SideNote from '../../UI/SideNote/SideNote';

const BeamSearchSteps = () => {
  return (
    <div>
      <p className="body-text">
        To identify the best matches at each step, <b>beam search</b> is used, a
        decoding strategy that, at each step, considers multiple possible
        continuations of a sequence based on the probabilities predicted by an
        autoregressive model, and keeps only the most promising candidates for
        further extension.
        <br />
        <br />
        Let’s walk through a toy example of beam search on a single‐nucleotide
        sequence. Starting with the prompt <i>“A,”</i> we perform beam search
        over the alphabet <b>[A, T, C, G]</b>. At each time step we expand three
        candidate nucleotides
        <SideNote>
          Also known as <i>hypotheses</i>.
        </SideNote>
        , then keep only the two highest‑scoring beams (beam size=2).
        <br />
        <br />
        Then, at T = 1 (the first step), we have three hypotheses: A, T, and C.
        We extend each of these hypotheses with every possible next character,
        and retain only the top-scoring two candidates to continue to the next
        step, <b>A</b> and <b>C</b>, with new sequences <b>AA</b> and <b>AC</b>:
      </p>
      <div style={{ margin: '2rem auto', maxWidth: '900px' }}>
        <BeamSearch1 />
      </div>
      <p className="body-text">
        At time T = 2, we extend each of these hypotheses with every possible
        next character, and retain only the top-scoring ones to continue to the
        next step, <b>AAG</b> and <b>ACC</b>:
      </p>
      <div style={{ margin: '2rem auto', maxWidth: '900px' }}>
        <BeamSearch2 />
      </div>
      <p className="body-text">We continue the process for T = 3.</p>
      <div style={{ margin: '2rem auto', maxWidth: '900px' }}>
        <BeamSearch3 />
      </div>
      <p className="body-text">
        Continuing the process, at time T = 4, we finalize our selection of{' '}
        <b>AAGG</b> as the best sequence.
      </p>
      <div style={{ margin: '2rem auto', maxWidth: '900px' }}>
        <BeamSearch4 />
      </div>
      <p className="body-text">
        Of course, the above was a simplified example. Let's see how beam search
        is applied to the genome generation task.
      </p>
    </div>
  );
};

export default BeamSearchSteps;
