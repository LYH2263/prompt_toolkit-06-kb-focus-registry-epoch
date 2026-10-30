"""
Acceptance tests for host-facing layout-focus / key-bindings registry
generations (epochs).

These tests exercise *timing causality* with ``Application.run_async`` and a
``PipeInput`` (no real terminal, no network):

* focus entering a modal truncates ancestor *normal* bindings while true
  global bindings stay alive (measurable fork);
* a multi-key prefix waiting for its timeout flush follows focus moves,
  global-activity flips and key-bindings reference replacements;
* replacing the Application level key-bindings reference stops the old
  handlers from being hit;
* inner/outer generations never share cache state.
"""

from __future__ import annotations

import asyncio
import time

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding.key_bindings import (
    ConditionalKeyBindings,
    GlobalOnlyKeyBindings,
    KeyBindings,
    RegistryGeneration,
    merge_key_bindings,
)
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import DummyOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window() -> Window:
    "Focusable window WITHOUT a BufferControl, so no default self-insert applies."
    return Window(FormattedTextControl(text="x", focusable=True))


def _modal_layout(kb_outer, kb_modal=None):
    """
    Root container (carrying ``kb_outer``) with a modal float (carrying
    ``kb_modal``)::

        FloatContainer(key_bindings=kb_outer)
        +-- w_main
        +-- Float: HSplit(modal=True, key_bindings=kb_modal)
                   +-- w_dialog
    """
    w_main = _window()
    w_dialog = _window()
    modal = HSplit([w_dialog], modal=True, key_bindings=kb_modal)
    root = FloatContainer(
        content=w_main,
        floats=[Float(content=modal)],
        key_bindings=kb_outer,
    )
    return root, w_main, w_dialog


def _prefix_focus_app(kb_outer, kb_modal, initial, inp):
    root, w_main, w_dialog = _modal_layout(kb_outer, kb_modal)
    initial_window = w_dialog if initial == "dialog" else w_main
    return Application(
        layout=Layout(root, focused_element=initial_window),
        input=inp,
        output=DummyOutput(),
    ), root, w_main, w_dialog


async def _drive(app, driver):
    "Drive an ``Application.run_async`` run with a driver coroutine."
    task = asyncio.create_task(app.run_async(set_exception_handler=False))
    dtask = asyncio.create_task(driver())
    await asyncio.wait_for(task, timeout=10)
    await asyncio.wait_for(dtask, timeout=10)


def _run(app, driver):
    asyncio.run(_drive(app, driver))


# ---------------------------------------------------------------------------
# 1. Modal fork: ancestor normals truncated, globals stay alive.
# ---------------------------------------------------------------------------


def test_focus_entering_modal_forks_normal_and_global_bindings():
    calls: list[str] = []

    kb_outer = KeyBindings()
    kb_outer.add("a")(lambda event: calls.append("ancestor-normal"))
    kb_outer.add("g", is_global=True)(
        lambda event: calls.append("ancestor-global")
    )

    root, w_main, w_dialog = _modal_layout(kb_outer)

    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(root, focused_element=w_main),
            input=inp,
            output=DummyOutput(),
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)

            # Focused in the outer pane: the normal ancestor binding hits.
            inp.send_text("a")
            await asyncio.sleep(0.1)

            # Focus migrates INTO the modal (generation boundary).
            app.layout.focus(w_dialog)
            await asyncio.sleep(0.05)

            # The ancestor NORMAL binding must be truncated now.
            inp.send_text("a")
            await asyncio.sleep(0.1)

            # But a true GLOBAL binding of the ancestor must stay alive.
            inp.send_text("g")
            await asyncio.sleep(0.1)

            app.exit()

        _run(app, driver)

    assert calls == ["ancestor-normal", "ancestor-global"]


def test_nested_modal_truncates_ancestor_normals_keeps_all_globals():
    """
    Focus inside an INNER modal: normals of both the root and the outer modal
    are truncated, normals of the inner modal hit, and true globals attached
    on EITHER ancestor level (root and outer modal subtree) stay alive.
    """
    calls: list[str] = []

    kb_outer = KeyBindings()
    kb_outer.add("a")(lambda event: calls.append("root-normal"))
    kb_outer.add("g", is_global=True)(
        lambda event: calls.append("root-global")
    )
    kb_m1 = KeyBindings()
    kb_m1.add("b")(lambda event: calls.append("outer-modal-normal"))
    kb_m1.add("G", is_global=True)(
        lambda event: calls.append("outer-modal-global")
    )
    kb_m2 = KeyBindings()
    kb_m2.add("c")(lambda event: calls.append("inner-modal-normal"))

    w_main = _window()
    w_m1 = _window()
    w_m2 = _window()
    inner_fc = FloatContainer(
        content=w_m1,
        floats=[
            Float(content=HSplit([w_m2], modal=True, key_bindings=kb_m2))
        ],
    )
    outer_modal = HSplit([inner_fc], modal=True, key_bindings=kb_m1)
    root = FloatContainer(
        content=w_main,
        floats=[Float(content=outer_modal)],
        key_bindings=kb_outer,
    )

    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(root, focused_element=w_m2),  # deepest modal
            input=inp,
            output=DummyOutput(),
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            for key in ("c", "b", "a", "G", "g"):
                inp.send_text(key)
                await asyncio.sleep(0.08)
            app.exit()

        _run(app, driver)

    assert calls == [
        "inner-modal-normal",
        "outer-modal-global",
        "root-global",
    ]


# ---------------------------------------------------------------------------
# 2. Multi-key prefix waits follow changes before the timeout flush.
# ---------------------------------------------------------------------------


def test_prefix_wait_flush_without_focus_switch_hits_dialog_binding():
    "Control: without a change during the wait, the dialog binding is flushed."
    calls: list[str] = []

    kb_outer = KeyBindings()
    kb_outer.add("x")(lambda event: calls.append("outer-x"))
    kb_outer.add("x", "y")(lambda event: calls.append("outer-xy"))
    kb_modal = KeyBindings()
    kb_modal.add("x")(lambda event: calls.append("dialog-x"))
    kb_modal.add("x", "y")(lambda event: calls.append("dialog-xy"))

    with create_pipe_input() as inp:
        app, root, w_main, w_dialog = _prefix_focus_app(
            kb_outer, kb_modal, "dialog", inp
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            inp.send_text("x")  # prefix of x y in the dialog -> waits
            await asyncio.sleep(0.4)  # past the timeout flush
            app.exit()

        _run(app, driver)

    assert calls == ["dialog-x"]


def test_prefix_wait_focus_switch_flush_moves_to_new_pane():
    "Switching focus mid-prefix makes the flush resolve against the new pane."
    calls: list[str] = []

    kb_outer = KeyBindings()
    kb_outer.add("x")(lambda event: calls.append("outer-x"))
    kb_outer.add("x", "y")(lambda event: calls.append("outer-xy"))
    kb_modal = KeyBindings()
    kb_modal.add("x")(lambda event: calls.append("dialog-x"))
    kb_modal.add("x", "y")(lambda event: calls.append("dialog-xy"))

    with create_pipe_input() as inp:
        app, root, w_main, w_dialog = _prefix_focus_app(
            kb_outer, kb_modal, "dialog", inp
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            inp.send_text("x")  # prefix while the dialog is focused
            await asyncio.sleep(0.05)
            app.layout.focus(w_main)  # leave the modal during the wait window
            await asyncio.sleep(0.4)  # let the timeout flush happen
            app.exit()

        _run(app, driver)

    # The new pane's exact binding is hit; the pre-change pane never fires.
    assert calls == ["outer-x"]


def test_prefix_wait_global_flip_without_generation_stays_snapshot():
    "Without a generation declaration, the global snapshot is kept (pin-like)."
    calls: list[str] = []
    flag = {"v": False}

    kb_outer = KeyBindings()
    kb_outer.add("g", is_global=Condition(lambda: flag["v"]))(
        lambda event: calls.append("ancestor-global")
    )
    kb_modal = KeyBindings()
    kb_modal.add("g", "z")(lambda event: calls.append("dialog-gz"))

    with create_pipe_input() as inp:
        app, root, w_main, w_dialog = _prefix_focus_app(
            kb_outer, kb_modal, "dialog", inp
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            inp.send_text("g")  # prefix of g z
            await asyncio.sleep(0.05)
            flag["v"] = True  # flip, but DO NOT declare a generation
            await asyncio.sleep(0.4)
            app.exit()

        _run(app, driver)

    assert calls == []


def test_prefix_wait_global_flip_with_generation_picks_up_global():
    "A generation boundary after a global flip lets the flush find it."
    calls: list[str] = []
    flag = {"v": False}

    kb_outer = KeyBindings()
    kb_outer.add("g", is_global=Condition(lambda: flag["v"]))(
        lambda event: calls.append("ancestor-global")
    )
    kb_modal = KeyBindings()
    kb_modal.add("g", "z")(lambda event: calls.append("dialog-gz"))

    with create_pipe_input() as inp:
        app, root, w_main, w_dialog = _prefix_focus_app(
            kb_outer, kb_modal, "dialog", inp
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            inp.send_text("g")  # prefix of g z while the dialog is focused
            await asyncio.sleep(0.05)
            flag["v"] = True
            app.bump_key_bindings_generation()  # declare the new ledger
            await asyncio.sleep(0.4)  # timeout flush must resolve against it
            app.exit()

        _run(app, driver)

    assert calls == ["ancestor-global"]


def test_prefix_wait_key_bindings_reference_swap_during_wait():
    "Replacing the mounted registry mid-prefix flushes against the new table."
    calls: list[str] = []

    kb_old = KeyBindings()
    kb_old.add("w")(lambda event: calls.append("old"))
    kb_old.add("w", "2")(lambda event: calls.append("old-w2"))
    kb_new = KeyBindings()
    kb_new.add("w")(lambda event: calls.append("new"))
    kb_new.add("w", "2")(lambda event: calls.append("new-w2"))

    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(_window()),
            key_bindings=kb_old,
            input=inp,
            output=DummyOutput(),
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            inp.send_text("w")  # prefix of w 2 -> waits against the old table
            await asyncio.sleep(0.05)
            app.key_bindings = kb_new  # remount; declares a generation
            await asyncio.sleep(0.4)  # flush must hit the new table
            app.exit()

        _run(app, driver)

    assert calls == ["new"]


def test_prefix_cut_over_is_immediate_when_focus_enters_modal():
    """
    The KeyProcessor watches the generation during a prefix wait: a focus
    migration must cut the pending sequence over to the new ledger without
    waiting for the full `timeoutlen`.
    """
    timings: list[tuple[str, float]] = []

    kb_outer = KeyBindings()
    kb_outer.add("z", is_global=True)(
        lambda event: timings.append(("global-z", time.monotonic()))
    )
    kb_modal = KeyBindings()
    kb_modal.add("z", "9")(lambda event: timings.append(("dialog-z9",)))

    with create_pipe_input() as inp:
        app, root, w_main, w_dialog = _prefix_focus_app(
            kb_outer, kb_modal, "dialog", inp
        )
        app.timeoutlen = 0.5  # much longer than the cut-over delay

        t0 = {"v": 0.0}

        async def driver():
            await asyncio.sleep(0.2)
            t0["v"] = time.monotonic()
            inp.send_text("z")  # prefix of z 9 in the dialog
            await asyncio.sleep(0.05)
            app.layout.focus(w_main)  # generation bump during the wait
            await asyncio.sleep(0.25)
            app.exit()

        _run(app, driver)

    assert [name for name, _ in timings] == ["global-z"]
    delay = timings[0][1] - t0["v"]
    # Cut-over happens on the next loop iteration, not after 0.5s.
    assert delay < 0.3, delay


# ---------------------------------------------------------------------------
# 3. Replacing the Application key-bindings reference.
# ---------------------------------------------------------------------------


def test_replace_application_key_bindings_stops_old_handlers():
    calls: list[str] = []

    kb_old = KeyBindings()
    kb_old.add("h")(lambda event: calls.append("old"))
    kb_new = KeyBindings()
    kb_new.add("h")(lambda event: calls.append("new"))

    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(_window()),
            key_bindings=kb_old,
            input=inp,
            output=DummyOutput(),
        )
        app.timeoutlen = 0.15

        async def driver():
            await asyncio.sleep(0.15)
            inp.send_text("h")  # populates the focus merge with the old table
            await asyncio.sleep(0.15)
            assert calls == ["old"]

            epoch_before = app.registry_generation.epoch
            app.key_bindings = kb_new  # remount: implicit generation boundary
            assert app.registry_generation.epoch > epoch_before

            inp.send_text("h")
            await asyncio.sleep(0.2)
            app.exit()

        _run(app, driver)

    assert calls == ["old", "new"]


def test_assign_same_key_bindings_reference_does_not_bump():
    with create_pipe_input() as inp:
        kb = KeyBindings()
        app = Application(
            layout=Layout(_window()),
            key_bindings=kb,
            input=inp,
            output=DummyOutput(),
        )
        epoch = app.registry_generation.epoch
        app.key_bindings = kb
        assert app.registry_generation.epoch == epoch


# ---------------------------------------------------------------------------
# 4. Inner/outer generations never share cache state.
# ---------------------------------------------------------------------------


def test_same_application_second_run_never_serves_stale_merge():
    """
    A second (inner) run on the same Application must not eat the merge cached
    for the first run after the key-bindings reference was replaced.
    """
    calls: list[str] = []

    kb_old = KeyBindings()
    kb_old.add("s")(lambda event: calls.append("old"))
    kb_new = KeyBindings()
    kb_new.add("s")(lambda event: calls.append("new"))

    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(_window()),
            key_bindings=kb_old,
            input=inp,
            output=DummyOutput(),
        )
        app.timeoutlen = 0.15

        async def driver_first():
            await asyncio.sleep(0.15)
            inp.send_text("s")
            await asyncio.sleep(0.2)
            app.exit()

        async def driver_second():
            await asyncio.sleep(0.15)
            inp.send_text("s")
            await asyncio.sleep(0.2)
            app.exit()

        async def both_runs():
            await _drive(app, driver_first)
            assert calls == ["old"]

            # Replace the mounted registry between the two runs.
            app.key_bindings = kb_new

            await _drive(app, driver_second)

        asyncio.run(both_runs())

    assert calls == ["old", "new"]


def test_inner_application_generation_does_not_touch_outer_cache():
    """
    A complete inner application run (with focus migrations of its own) must
    not advance the outer Application's generation nor invalidate its cached
    focus merge.
    """
    outer_calls: list[str] = []
    inner_calls: list[str] = []

    kb_outer = KeyBindings()
    kb_outer.add("h")(lambda event: outer_calls.append("outer"))
    kb_inner = KeyBindings()
    kb_inner.add("h")(lambda event: inner_calls.append("inner"))

    w_o1 = _window()
    w_o2 = _window()
    root_o = FloatContainer(
        content=w_o1,
        floats=[Float(content=w_o2)],
        key_bindings=kb_outer,
    )

    w_i1 = _window()
    w_i2 = _window()
    root_i = FloatContainer(
        content=w_i1,
        floats=[Float(content=w_i2)],
        key_bindings=kb_inner,
    )

    with create_pipe_input() as inp_o, create_pipe_input() as inp_i:
        app_outer = Application(
            layout=Layout(root_o, focused_element=w_o1),
            input=inp_o,
            output=DummyOutput(),
        )
        app_inner = Application(
            layout=Layout(root_i, focused_element=w_i1),
            input=inp_i,
            output=DummyOutput(),
        )
        app_outer.timeoutlen = 0.15
        app_inner.timeoutlen = 0.15

        combined = app_outer.key_processor._bindings

        async def driver():
            await asyncio.sleep(0.15)

            # Steady state for the outer app.
            inp_o.send_text("h")
            await asyncio.sleep(0.15)
            outer_epoch = app_outer.registry_generation.epoch
            outer_merge = combined._key_bindings

            # Run the inner application to completion, moving its focus.
            inner_task = asyncio.create_task(
                app_inner.run_async(set_exception_handler=False)
            )
            await asyncio.sleep(0.15)
            app_inner.layout.focus(w_i2)  # bumps the INNER generation only
            await asyncio.sleep(0.05)
            inp_i.send_text("h")
            await asyncio.sleep(0.15)
            app_inner.exit()
            await asyncio.wait_for(inner_task, timeout=10)

            # The outer ledger is untouched.
            assert app_outer.registry_generation.epoch == outer_epoch
            assert combined._key_bindings is outer_merge
            assert inner_calls == ["inner"]

            inp_o.send_text("h")
            await asyncio.sleep(0.15)
            app_outer.exit()

        _run(app_outer, driver)

    assert outer_calls == ["outer", "outer"]


# ---------------------------------------------------------------------------
# 5. RegistryGeneration + proxies unit behaviour.
# ---------------------------------------------------------------------------


def test_registry_generation_monotonic_and_listeners():
    generation = RegistryGeneration()
    assert generation.epoch == 0

    seen: list[int] = []
    unsubscribe = generation.subscribe(seen.append)

    assert generation.bump() == 1
    assert generation.epoch == 1
    generation.bump()
    assert seen == [1, 2]

    unsubscribe()
    generation.bump()
    assert seen == [1, 2]


def test_global_only_without_generation_keeps_snapshot():
    "Unbound proxy: `is_global` flips are NOT picked up without a version change."
    flag = {"v": False}

    kb = KeyBindings()
    kb.add("a", is_global=Condition(lambda: flag["v"]))(lambda event: None)

    only = GlobalOnlyKeyBindings(kb)
    assert only.get_bindings_for_keys(("a",)) == []

    flag["v"] = True  # no child version change, no generation -> still hidden
    assert only.get_bindings_for_keys(("a",)) == []


def test_global_only_with_generation_resnapshots_on_bump():
    flag = {"v": False}

    kb = KeyBindings()
    kb.add("a", is_global=Condition(lambda: flag["v"]))(lambda event: None)

    generation = RegistryGeneration()
    only = GlobalOnlyKeyBindings(kb, generation=generation)
    assert only.get_bindings_for_keys(("a",)) == []

    flag["v"] = True
    generation.bump()
    bindings = only.get_bindings_for_keys(("a",))
    assert len(bindings) == 1
    assert bindings[0].is_global() is True


def test_conditional_bindings_rebuild_on_generation_and_filter_live():
    flag = {"v": False}
    outer = {"on": True}

    kb = KeyBindings()
    kb.add("a", filter=Condition(lambda: flag["v"]))(lambda event: None)

    generation = RegistryGeneration()
    cond = ConditionalKeyBindings(
        kb, Condition(lambda: outer["on"]), generation=generation
    )

    # The binding is present but inactive until its filter flips.
    assert cond.get_bindings_for_keys(("a",))[0].filter() is False
    flag["v"] = True
    # Filters are evaluated live at match time, even without a rebuild.
    assert cond.get_bindings_for_keys(("a",))[0].filter() is True

    # A generation boundary rebuilds the copies (still one binding).
    generation.bump()
    assert len(cond.get_bindings_for_keys(("a",))) == 1


def test_merge_key_bindings_rereads_on_generation():
    generation = RegistryGeneration()
    merged = merge_key_bindings([], generation=generation)
    before = merged.bindings

    # Access after a boundary rebuilds the internal flattened copy.
    generation.bump()
    assert merged.bindings is not before


def test_layout_focus_generation_counts_migrations():
    w1 = _window()
    w2 = _window()
    layout = Layout(HSplit([w1, w2]))
    assert layout.focus_generation >= 0
    before = layout.focus_generation
    layout.focus(w2)
    assert layout.focus_generation == before + 1
    layout.focus_last()
    assert layout.focus_generation == before + 2


def test_application_has_generation_without_new_constructor_arguments():
    "No new required constructor parameters; the generation just exists."
    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(_window()),
            input=inp,
            output=DummyOutput(),
        )
        assert isinstance(app.registry_generation, RegistryGeneration)
        assert app.key_processor._bindings.generation is app.registry_generation


def test_standalone_key_processor_has_no_generation_path():
    """
    A KeyProcessor over a plain KeyBindings (no mounted Application
    generation) still drives a pending timeout without a generation watcher.
    """
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.key_binding.key_processor import KeyPress, KeyProcessor
    from prompt_toolkit.keys import Keys

    kb = KeyBindings()
    seen: list[str] = []
    kb.add(Keys.ControlX, Keys.ControlC)(lambda event: seen.append("xc"))

    def create_background_task(coroutine, **kw):
        coroutine.close()
        return None

    with create_pipe_input() as inp:
        app = Application(
            layout=Layout(_window()),
            input=inp,
            output=DummyOutput(),
        )
        # Don't start background tasks: a plain KeyBindings has no generation
        # watcher either, so the key pump behaves exactly as without epochs.
        app.create_background_task = create_background_task  # type: ignore[method-assign]
        with set_app(app):
            processor = KeyProcessor(kb)
            assert getattr(processor._bindings, "generation", None) is None
            processor.feed(KeyPress(Keys.ControlX, "\x18"))
            processor.feed(KeyPress(Keys.ControlC, "\x03"))
            processor.process_keys()
            assert seen == ["xc"]
