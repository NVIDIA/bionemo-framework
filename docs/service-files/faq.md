# Frequently Asked Questions (FAQ)

1. How do I submit an asynchronous request for a command and fetch the result?

    Here is an example python implementation to retrieve results from an OpenFold prediction, which only accepts async requests, and then write those results to a file.

**Python (via requests library)**

```python
import requests
import json
import time

API_HOST = "https://api.bionemo.ngc.nvidia.com/v1"
API_KEY = "NGC_API_KEY"

response = requests.post(
    f"{API_HOST}/protein-structure/alphafold2/predict",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files={
        "sequence": (None, "MAGVKALVALSFSGAIGLTFLMLGCALEDYGVYWP"),
        "relax_prediction": (None, "true"),
        "use_msa": (None, "true"),
    }

)
result = json.loads(response.content)
request_id = result["correlation_id"]


while True:
    response = requests.get(
        f"{API_HOST}/task/{request_id}",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    status_result = json.loads(response.content)
    result = status_result['control_info']
    if result['status'] in ['DONE', 'CANCEL', 'ERROR']:
        break
    else:
        time.sleep(20)

if result['status'] == 'DONE':
    filename = request_id.split('-')[-1]
    with open(f"{filename}.pdb", 'w') as pdb_file:
        pdb_file.write(status_result['response'])
    print(f'Job succeeded. File saved to {filename}.pdb')
else:
    print(f'Job ended with status {result["status"]}')
```

2. When clicking on "Provide Feedback" or "Report Bug," an empty browser window opens. How is this issue fixed?

    Feedback through email may not work as expected if the client machine does not have a properly configured default email client. The application may open a blank browser tab or an incorrect application instead of an email compose window. Follow these instructions to fix the issue.

    **Windows:**

    Follow the instructions at [https://support.microsoft.com/en-us/windows/change-default-programs-in-windows-e5d82cad-17d1-c53b-3505-f10a32e1894d](https://support.microsoft.com/en-us/windows/change-default-programs-in-windows-e5d82cad-17d1-c53b-3505-f10a32e1894d) to select the preferred email client.

    **Mac:**

    Follow the instructions at
    [https://support.apple.com/en-us/HT201607](https://support.apple.com/en-us/HT201607) to select the preferred email client.

    **Linux:**

    Follow the instructions at [https://askubuntu.com/questions/636527/default-program-selection](https://askubuntu.com/questions/636527/default-program-selection) to select the preferred email client.


3. Why are portions of the protein structure shown in the viewer fragmented?

    If relaxation was not performed on the predicted PDB structure (`"relax_prediction"="true"`), there can be structural violations present which can cause gaps to appear when the structure is visualized.
