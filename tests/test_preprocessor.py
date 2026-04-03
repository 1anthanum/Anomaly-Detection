"""Tests for the data preprocessor."""

import numpy as np

from src.data.preprocessor import TimeSeriesWindower, WindowDataset, create_dataloader


def test_fit_and_normalize():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    windower = TimeSeriesWindower(window_size=3)
    windower.fit(data)

    normalized = windower.normalize(data)
    assert abs(normalized.mean()) < 1e-5
    assert abs(normalized.std() - 1.0) < 0.1


def test_denormalize_roundtrip():
    data = np.random.randn(100) * 10 + 50
    windower = TimeSeriesWindower(window_size=10)
    windower.fit(data)

    normalized = windower.normalize(data)
    recovered = windower.denormalize(normalized)
    np.testing.assert_allclose(recovered, data, atol=1e-6)


def test_create_windows():
    data = np.arange(10, dtype=float)
    windower = TimeSeriesWindower(window_size=3, stride=1)
    windower.fit(data)
    windows = windower.create_windows(data)
    assert windows.shape == (8, 3)  # 10 - 3 + 1 = 8 windows
    np.testing.assert_array_equal(windows[0], [0, 1, 2])
    np.testing.assert_array_equal(windows[-1], [7, 8, 9])


def test_create_windows_with_stride():
    data = np.arange(10, dtype=float)
    windower = TimeSeriesWindower(window_size=3, stride=3)
    windower.fit(data)
    windows = windower.create_windows(data)
    assert windows.shape[0] == 3  # ceil((10-3+1)/3) windows


def test_prepare():
    data = np.random.randn(100)
    windower = TimeSeriesWindower(window_size=10)
    windows = windower.prepare(data, fit=True)
    assert windows.shape == (91, 10)


def test_window_dataset():
    windows = np.random.randn(50, 10)
    dataset = WindowDataset(windows)
    assert len(dataset) == 50
    x, y = dataset[0]
    assert x.shape == (10, 1)
    assert (x == y).all()  # autoencoder: input == target


def test_create_dataloader():
    windows = np.random.randn(50, 10)
    loader = create_dataloader(windows, batch_size=16, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (16, 10, 1)


if __name__ == "__main__":
    test_fit_and_normalize()
    test_denormalize_roundtrip()
    test_create_windows()
    test_create_windows_with_stride()
    test_prepare()
    test_window_dataset()
    test_create_dataloader()
    print("All preprocessor tests passed!")
