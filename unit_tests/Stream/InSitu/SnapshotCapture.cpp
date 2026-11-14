//
// Unit-test support: Catalyst field-snapshot capture implementation
//
// Copies each execute payload and returns it through catalyst_results so the channel
// tests can inspect the actual Conduit fields and values received by the backend.
//

#include <catalyst.h>
#include <catalyst_conduit.h>

#include "catalyst_impl_ippl_snapshot_capture.h"

namespace {
conduit_node* snapshot = nullptr;
}

catalyst_status catalyst_initialize_ippl_snapshot_capture(const conduit_node*) {
    snapshot = conduit_node_create();
    return catalyst_status_ok;
}

catalyst_status catalyst_execute_ippl_snapshot_capture(const conduit_node* params) {
    conduit_node_set_node(snapshot, const_cast<conduit_node*>(params));
    return catalyst_status_ok;
}

catalyst_status catalyst_results_ippl_snapshot_capture(conduit_node* params) {
    conduit_node_set_node(params, snapshot);
    return catalyst_status_ok;
}

catalyst_status catalyst_finalize_ippl_snapshot_capture(const conduit_node*) {
    conduit_node_destroy(snapshot);
    snapshot = nullptr;
    return catalyst_status_ok;
}

catalyst_status catalyst_about_ippl_snapshot_capture(conduit_node* params) {
    conduit_node_set_path_char8_str(params, "catalyst/implementation", "ippl_snapshot_capture");
    return catalyst_status_ok;
}
