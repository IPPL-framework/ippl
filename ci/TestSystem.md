# `TestResult` Specification

This section defines the structure of a `TestResult` — the symbolic representation of a single test execution outcome in the IPPL testing infrastructure. This result object is used internally by the test runner and reporting infrastructure and is designed to be serialized (e.g., to JSON) and transformed into formats such as Allure result files.

The structure is designed to:
- Be flexible by having only a small number of required field.
- Support a rich set of features through optional arguments.

---

## Required Fields

These fields must be present in all `TestResult` objects.

| Field       | Type            | Description                                                                                                         |
| ----------- | --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `name`      | `string`        | Unique short name for the specific test in a specific environment                                                   |
| `status`    | `string`        | Outcome of the test: `"passed"`, `"failed"`, `"skipped"`, or `"broken"`                                             |
| `uuid`      | `string (uuid)` | Unique identifier for this specific test execution. Must be different each time the test runs.                      |
| `testRunId` | `string`        | ID shared across all tests in a single run/session. Used to group results for a test batch that are shown together. |
| `historyId` | `string`        | Logical stable ID of the test, used to identify a test through different runs.                                      |

---

## Optional Timing Fields

| Field      | Type   | Description |
|------------|--------|-------------|
| `start`    | `int`  | UNIX timestamp in milliseconds for when the test started |
| `stop`     | `int`  | UNIX timestamp in milliseconds for when the test ended |
| `duration` | `int`  | Test duration in milliseconds (useful when `start`/`stop` are unknown) |

---

## Optional Metadata (via `labels`)

All categorical or grouping metadata is passed through the `labels` array, following [Allure](https://allurereport.org/docs/how-it-works-test-result-file/) conventions.

| Field    | Type           | Description                                                 |
| -------- | -------------- | ----------------------------------------------------------- |
| `labels` | `list<object>` | Arbitrary metadata as `{ "name": <key>, "value": <value> }` |

---

## Other Optional Fields

| Field           | Type           | Description                                                 |
| --------------- | -------------- | ----------------------------------------------------------- |
| `parameters`    | `list<object>` | Input parameters (e.g., `"grid_size": "128x128x128"`)       |
| `steps`         | `list<object>` | Structured breakdown of test phases, with timing and status |
| `attachments`   | `list<object>` | Files related to the test (logs, images, raw output)        |
| `statusDetails` | `object`       | Contains `message` and `trace` for diagnostics              |
| `executor`      | `object`       | CI metadata (runner name, build URL, etc.)                  |
