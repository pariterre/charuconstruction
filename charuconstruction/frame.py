import cv2
import numpy as np


class Frame:
    def __init__(self, frame: np.ndarray):
        self._frame = frame
        self._grayscale_frame = self._ensure_grayscale()

    def get(self, grayscale: bool = False) -> np.ndarray:
        """
        Get the underlying image array.

        Parameters:
            grayscale (bool): Whether to return the grayscale version of the frame.
        Returns:
            np.ndarray: The image array.
        """
        if grayscale:
            return self._grayscale_frame
        return self._frame

    def show(
        self,
        max_width=1280,
        max_height=720,
        grayscale: bool = False,
        wait_time: int = None,
    ) -> bool:
        """
        Show the frame in a window, resized to fit within max dimensions.

        Parameters:
            max_width (int): Maximum width of the displayed window.
            max_height (int): Maximum height of the displayed window.
            grayscale (bool): Whether to show the grayscale version of the frame.
            wait_time (int): Time in milliseconds to wait for a key event. If None, waits indefinitely.
        Returns:
            bool: True if the window is visible, False if it is closed.
        """
        to_show = self._grayscale_frame if grayscale else self._frame

        h, w = to_show.shape[:2]

        # Compute scaling factor to fit max size
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)  # never upscale

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(to_show, (new_w, new_h))
        else:
            resized = to_show

        cv2.imshow("frame", resized)
        cv2.waitKey(wait_time)

        try:
            is_visible = (
                cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) >= 1
            )
            return is_visible

        except cv2.error:
            # Window was closed
            return False

    def _ensure_grayscale(self) -> np.ndarray:
        """Ensure the frame is in grayscale format."""
        if len(self._frame.shape) == 2:
            return self._frame
        elif len(self._frame.shape) == 3:
            return cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Invalid image shape")
