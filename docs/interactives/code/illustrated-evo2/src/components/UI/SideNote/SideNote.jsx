import React, { useRef, useEffect } from 'react';
import './SideNote.css';

const SideNote = ({ children }) => {
  const noteRef = useRef(null);
  const sideNoteRef = useRef(null);

  useEffect(() => {
    const positionSideNote = () => {
      if (noteRef.current && sideNoteRef.current) {
        const referenceRect = noteRef.current.getBoundingClientRect();
        const container = noteRef.current.closest('.docsBody');
        const containerRect = container && container.getBoundingClientRect();

        if (containerRect) {
          const topPosition = referenceRect.top - containerRect.top;
          sideNoteRef.current.style.top = `${topPosition}px`;
        }
      }
    };

    // Position the sidenote initially and on resize
    positionSideNote();
    window.addEventListener('resize', positionSideNote);
    return () => window.removeEventListener('resize', positionSideNote);
  }, []);

  return (
    <>
      <span className="sidenote" ref={sideNoteRef}>
        {children}
      </span>
      <span className="reference" ref={noteRef}>
        <sup className="marker">*</sup>
      </span>
    </>
  );
};

export default SideNote;
