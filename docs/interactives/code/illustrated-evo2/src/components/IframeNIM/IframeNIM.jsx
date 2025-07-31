import { useState } from 'react';

const IframeNIM = () => {
  const nvidiaBuildUrl = 'https://build.nvidia.com/arc/evo2-40b';
  const [tryIframe, setTryIframe] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  const handleIframeLoad = () => {
    setIsLoading(false);
  };

  const handleIframeError = () => {
    setTryIframe(false);
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col w-full h-screen p-6">
      {tryIframe ? (
        <>
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-80 z-10">
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                <p className="text-gray-700 font-medium">
                  Attempting to load NVIDIA content...
                </p>
              </div>
            </div>
          )}

          <iframe
            src={nvidiaBuildUrl}
            className="w-full h-full border-0"
            onLoad={handleIframeLoad}
            onError={handleIframeError}
            title="NVIDIA Arc Evo2-40B"
            sandbox="allow-scripts allow-same-origin allow-forms"
            referrerPolicy="no-referrer"
          />
        </>
      ) : (
        <div className="flex flex-col items-center justify-center bg-white rounded-lg shadow-md p-8 max-w-2xl mx-auto">
          <div className="text-red-500 mb-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-12 w-12"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-800 mb-4">
            Unable to Display NVIDIA Content
          </h2>
          <p className="text-gray-600 mb-6 text-center">
            This content cannot be embedded in an iframe due to Content Security
            Policy restrictions set by NVIDIA.
          </p>
          <p className="text-gray-600 mb-6 text-center">
            The website's CSP only allows embedding on nvidia.com domains and
            specific partners.
          </p>
          <a
            href={nvidiaBuildUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="px-6 py-3 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 transition duration-300"
          >
            Visit NVIDIA Website Directly
          </a>
        </div>
      )}
    </div>
  );
};

export default IframeNIM;
