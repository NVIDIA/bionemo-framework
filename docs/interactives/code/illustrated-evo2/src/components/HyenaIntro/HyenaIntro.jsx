import katexify from '../../utils/katexify';
import './HyenaIntro.css';
import SideNote from '../UI/SideNote/SideNote';
import ToeplitzConvolutionVisualization from '../ToeplitzMatrix/ToeplitzConvolutionVisualization';
import HyenaOperatorDiagram from '../HyenaOperatorDiagram/HyenaOperatorDiagram';
import HyenaSideBySide from './HyenaSideBySide';
import HyenaOperators from '../HyenaOperators/HyenaOperators';
import HyenaLongContext from './HyenaLongContext';
import HyenaDataDependence from './HyenaDataDependence';

const HyenaIntro = () => {
  const math4 = `\\begin{aligned}
  q_t^\\alpha &= T_{t t'}^\\alpha\\bigl(x_{t'}^\\beta,\\,W^{\\beta\\alpha}\\bigr),\\\\
  k_t^\\alpha &= H_{t t'}^\\alpha\\bigl(x_{t'}^\\beta,\\,U^{\\beta\\alpha}\\bigr),\\\\
  v_t^\\alpha &= K_{t t'}^\\alpha\\bigl(x_{t'}^\\beta,\\,P^{\\beta\\alpha}\\bigr),\\\\
  y_t^\\alpha &= \\bigl(q_t^\\beta,\\,G_{t t'}^\\beta,\\,k_{t'}^\\beta,\\,v_{t'}^\\beta\\bigr)\\,M^{\\beta\\alpha}.
  \\end{aligned}`;

  const math = text => (
    <span dangerouslySetInnerHTML={{ __html: katexify(text, false) }} />
  );

  return (
    <>
      <h1 className="body-header">Hyena Operators</h1>

      <p className="body-text">
        Of course, a core component of the StripedHyena2 architecture is, as the
        name implies, the Hyena operator. But what is a Hyena operator?
        <br />
        <br />
        Hyena Operators
        <a id="citation-4" href="#ref-4" className="citation-link">
          <sup>4</sup>
        </a>{' '}
        are a class of data-controlled operators consisting of a recurrence of
        multiplicative gating interactions and interleaved convolutions. They
        are designed to be as expressive as attention but with reduced
        computational cost. At a high level, the operator was designed to
        capture two key aspects of attention:
        <br />
        <br />
        <ol style={{ paddingLeft: '1.5em' }}>
          <li>
            <strong>Global Context:</strong> Attention provides a global
            receptive field: it compares every value to each other value, so it
            is able to capture long-range dependencies. In the Hyena operator,
            this global context is achieved via long implicit convolutions, at a
            subquadratic cost.
          </li>
          <br />
          <li>
            <strong>Data-dependence:</strong> Attention is able to capture
            data-dependent relationships; as we go through the autoregressive
            sequence generation, the context weights themselves change. In the
            Hyena operator, this is achieved via an element-wise 'gating'
            multiplication that occurs throughout the operator.
          </li>
        </ol>
        <br />
        The StripedHyena2 architecture comprises three different Hyena
        operators: Hyena Short Explicit (<b>Hyena SE</b>), Hyena Medium
        Regularized (<b>Hyena MR</b>), and Hyena Long Implicit (<b>Hyena LI</b>
        ):
      </p>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <HyenaSideBySide />
      </div>
      <p className="body-text">
        As visible in the diagram above, all Hyena operators follow the same
        structure of an initial dense projection and short convolution. What
        differentiates them is the inner filter, the convolution kernel used in
        between the elementwise gating interactions.
      </p>
      <HyenaOperators />

      <HyenaOperatorDiagram />

      <p className="body-text">
        Note that, just as attention computes three linear projections—q, k, and
        v—from the input sequence, Hyena applies N+1 linear projections to its
        input. In Evo 2, N+1 is set to 3, which conveniently matches the number
        of projections used in attention. However, Hyena is more flexible: you
        can choose any number of projections, for a <i>Hyena Recurrence</i> of
        arbitrary depth.
        <br />
        <br />
        Then, our input-output function can be simply defined as{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(`y = H(u)v`, false),
          }}
        />
        , where{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(`H(u)`, false),
          }}
        />{' '}
        is defined by interleaving convolutions and element-wise multiplication
        with one projection{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(`x_i`, false),
          }}
        />{' '}
        one at a time, until every projection has been used. For the specific
        case where N+1 = 3, interpreting the projections as q, k, and v, the
        process is mathematically described as follows:
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(math4, true),
          }}
        />
        where{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(
              `T,\\,H,\\,K,\\,G\\in\\mathbb{R}^{d\\times\\ell\\times\\ell}`,
              false
            ),
          }}
        />{' '}
        are Toeplitz matrices (corresponding to the convolution with the
        respective filters{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(`h_T,\\,h_H,\\,h_K,\\,h_G`, false),
          }}
        />
        ), and{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(
              `W,\\,U,\\,P,\\,M\\in\\mathbb{R}^{d\\times d}`,
              false
            ),
          }}
        />{' '}
        are dense matrices (parametrized as dense matrices or low‑rank
        matrices).
        <br />
        <br />A key insight from training is that not every input-dependent
        convolution in a hybrid architecture should rely on long, implicit
        filters. Following this insight, the filters {math('h_T')},{' '}
        {math('h_H')}, {math('h_K')} are parametrized explicitly: the entries of
        the filters are learnable parameters, analogous to the approach of
        classical convolutional neural networks. The inner filter {math('h_G')}{' '}
        for Hyena LI is instead parametrized implicitly.
        <br />
        <br />
        <b>Convolution as (Toeplitz) Matrix Multiplication</b>
        <br />A Toeplitz matrix represents a convolution mathematically; each
        row is a shifted copy of the filter and each diagonal from top left to
        bottom right has the same value. In practical terms, Toeplitz matrices
        are the mathematical structure underlying 1D convolutions. The
        multiplication of a Toeplitz matrix with a vector is equivalent to
        performing a convolution operation between a filter (the first
        row/column of the matrix) and an input sequence. Note that causal
        convolutions use lower triangular Toeplitz matrices where each output
        depends only on current and past inputs, not future ones.
        <br />
        <br />
        Explore both causal and non-causal convolutions with Toeplitz matrices
        by hovering over the <b>Padded Input</b> and <b>Output</b> below.
      </p>
      <ToeplitzConvolutionVisualization />
      <br />
      <br />
      <p className="body-text">
        Now that we have a better understanding of the Hyena operator, let's
        revisit the two core insights that led to its design,{' '}
        <b>Global Context</b> and <b>Data-dependence</b>.
      </p>
      <HyenaLongContext />
      <HyenaDataDependence />
    </>
  );
};

export default HyenaIntro;
