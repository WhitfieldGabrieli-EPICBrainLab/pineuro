"""Tests for pineuro.gui.components.task_mirror — the operator's Task Display.

Construction-only: no page is attached, so ``update_state``'s ``_image.update()``
raises and is swallowed by the component's own ``contextlib.suppress`` — the ``src``
assignment under test happens first either way.
"""

import flet as ft


class TestBuild:
    def test_returns_container_wrapping_an_image(self):
        from pineuro.gui.components.task_mirror import TaskMirror

        container = TaskMirror().build()
        assert isinstance(container, ft.Container)
        assert isinstance(container.content, ft.Image)

    def test_gapless_playback_is_enabled(self):
        """Regression guard: without this the mirror blanks between every frame.

        Flet's default is False, which "briefly shows nothing" when the image
        provider changes — at the ~15 Hz capture rate that reads as a strobe.
        """
        from pineuro.gui.components.task_mirror import TaskMirror

        assert TaskMirror().build().content.gapless_playback is True

    def test_defaults_to_the_task_mirror_style_tokens(self):
        from pineuro.gui import styles as st
        from pineuro.gui.components.task_mirror import TaskMirror

        container = TaskMirror().build()
        assert (container.width, container.height) == (st.TASK_MIRROR_W, st.TASK_MIRROR_H)
        assert (container.content.width, container.content.height) == (st.TASK_MIRROR_W, st.TASK_MIRROR_H)

    def test_explicit_dimensions_reach_both_image_and_container(self):
        from pineuro.gui.components.task_mirror import TaskMirror

        container = TaskMirror(width=320, height=180).build()
        assert (container.width, container.height) == (320, 180)
        assert (container.content.width, container.content.height) == (320, 180)


class TestUpdateState:
    def _built(self):
        from pineuro.gui.components.task_mirror import TaskMirror

        mirror = TaskMirror()
        mirror.build()
        return mirror

    def test_data_uri_is_used_verbatim(self):
        mirror = self._built()
        mirror.update_state({"type": "screenshot", "image": "data:image/jpeg;base64,QQ=="})
        assert mirror._image.src == "data:image/jpeg;base64,QQ=="

    def test_raw_base64_is_wrapped_as_png(self):
        mirror = self._built()
        mirror.update_state({"type": "screenshot", "image": "QQ=="})
        assert mirror._image.src == "data:image/png;base64,QQ=="

    def test_empty_image_leaves_src_unchanged(self):
        mirror = self._built()
        before = mirror._image.src
        mirror.update_state({"type": "screenshot", "image": ""})
        assert mirror._image.src == before

    def test_non_screenshot_state_leaves_src_unchanged(self):
        """Guards against a re-introduced vector-mirror path.

        The removed ``get_mirror_state()`` emitted this shape; it carries no
        ``type`` key and must never be rendered.
        """
        mirror = self._built()
        before = mirror._image.src
        mirror.update_state({"background": "black", "shapes": []})
        assert mirror._image.src == before

    def test_update_before_build_does_not_raise(self):
        from pineuro.gui.components.task_mirror import TaskMirror

        TaskMirror().update_state({"type": "screenshot", "image": "QQ=="})
