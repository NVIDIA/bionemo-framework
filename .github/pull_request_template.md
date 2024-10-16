(**NOTE:** _**delete** these instructional lines as you fill-out this PR template_)

(**NOTE:** _template is designed to be filled-in and used as the **squashed commit message for the entire PR**. _Italicized text_ is intended to be deleted as you fill in this template. Use the text between the `---`)

---

_High level summary of changes. Try to keep this as short and informative as possible: less is more._

_Describe your changes. You can be more detailed and descriptive here. If it is a code change, Be sure to answer:_
  - _What is changing?_
  - _What is the new or fixed functionality?_
  - _Why or when would someone want to use these changes?_
  - _How can someone use these changes?_
---

## Summary
_High level summary of changes. Try to keep this as short and informative as possible: less is more._

## Details
_Describe your changes. You can be more detailed and descriptive here._

## Usage
_How does a user interact with the changed code?_
```python
python -m your.new.module -and -all -options
```

## Testing
_How do you prove that your code behaves the way you claim?_

Tests for these changes can be run via:
```shell
pytest -v tests/your/new/or/existing/test_functions.py::test_function
```


(**NOTE:** _also **delete** this checklist as you fill-out this PR template_)

**Most of the changes** to files with extensions `*.py`, `*.yaml`, `*.yml`, `Dockerfile*` or `requirements.txt` **DO REQUIRE both `pytest-` and `jet-` CI stages**.

- [ ] Did you review the [Before your PR is "Ready for review" section](https://github.com/NVIDIA/bionemo-framework/-/blob/dev/CONTRIBUTING.md?ref_type=heads#before-pr-ready) before asking for review?
- [ ] Did you make sure your changes have tests? Did you test your changes locally?
- [ ] Can you add [the `SKIP_CI` label](https://github.com/NVIDIA/bionemo-framework/-/blob/dev/CONTRIBUTING.md?ref_type=heads#skip-ci) to your PR?
- [ ] Can you add [the `PYTEST_NOT_REQUIRED` label](https://github.com/NVIDIA/bionemo-framework/-/blob/dev/CONTRIBUTING.md?ref_type=heads#skip-pytest) to your PR?
- [ ] Can you add [the `JET_NOT_REQUIRED` label](https://github.com/NVIDIA/bionemo-framework/-/blob/dev/CONTRIBUTING.md?ref_type=heads#skip-jet) to your PR?
