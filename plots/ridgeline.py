import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde, iqr
from webcolors import hex_to_rgb, name_to_rgb


def _create_array_dict(data, category_col, data_col, normalize_each, scaling_factor):
    array_dict = {}
    target_area = (data[data_col].max() - data[data_col].min()) * 0.25
    max_y_normalized = 0

    for category in list(data[category_col].unique()):
        array_dict[f"x_{category}"] = data[data[category_col] == category][data_col]
        array_dict[f"y_{category}"] = data[data[category_col] == category]["count"]
        if normalize_each:
            current_area = np.trapz(
                array_dict[f"y_{category}"].abs(), array_dict[f"x_{category}"]
            )
            normalization_factor = target_area / current_area if current_area > 0 else 1
            normalized_y = array_dict[f"y_{category}"].abs() * normalization_factor
            array_dict[f"y_{category}_normalized"] = normalized_y * np.sign(
                array_dict[f"y_{category}"]
            )
            array_dict[f"{category}_norm_factor"] = normalization_factor
            max_y_normalized = max(max_y_normalized, normalized_y.max())

        else:
            min_count = data["count"].abs().min()
            max_count = data["count"].abs().max()
            array_dict[f"y_{category}_normalized"] = (
                (array_dict[f"y_{category}"].abs() - min_count)
                / (max_count - min_count)
                * scaling_factor
            ) * np.sign(array_dict[f"y_{category}"])
            array_dict[f"{category}_norm_factor"] = (
                scaling_factor / (max_count - min_count)
            ) - (min_count * scaling_factor / (max_count - min_count))
        sample_value = array_dict[f"x_{category}"].dropna().iloc[0]
        if isinstance(sample_value, (int | float | np.number)):
            array_dict[f"x_{category}"] = array_dict[f"x_{category}"].astype(float)
        elif isinstance(sample_value, (str | pd.Timestamp)):
            array_dict[f"x_{category}"] = pd.to_datetime(
                array_dict[f"x_{category}"], errors="coerce"
            )

    if normalize_each and max_y_normalized > 0:
        adjustment_factor = scaling_factor / max_y_normalized
        for category in list(data[category_col].unique()):
            array_dict[f"y_{category}_normalized"] *= adjustment_factor
            array_dict[f"{category}_norm_factor"] *= adjustment_factor

    return array_dict


def create_histogram(
    data: pd.DataFrame,
    category_col: str,
    data_col: str,
    weights_col: str | None = None,
    bin_width: float | str | None = None,
    n_bins: int | None = None,
) -> pd.DataFrame:
    data = data.dropna(subset=data_col).copy()

    if np.issubdtype(data[data_col].dtype, np.datetime64):
        data[data_col] = pd.to_datetime(data[data_col])

        if not bin_width and n_bins:
            # Calculate bin width based on n_bins for datetime data
            total_range = data[data_col].max() - data[data_col].min()
            bin_width = total_range / n_bins
            bins = pd.date_range(
                data[data_col].min() - bin_width,
                data[data_col].max() + 2 * bin_width,
                freq=bin_width,
            )
        elif bin_width:
            # Generate bins based on bin_width
            if isinstance(bin_width, str):
                bin_width = pd.to_timedelta(bin_width)
                bins = pd.date_range(
                    data[data_col].min() - bin_width,
                    data[data_col].max() + 2 * bin_width,
                    freq=bin_width,
                )
            else:
                bins = pd.date_range(
                    data[data_col].min() - bin_width,
                    data[data_col].max() + 2 * bin_width,
                    freq=bin_width,
                )
        else:
            # Default bin width: 1 day
            bin_width = "1D"
            bin_width = pd.to_timedelta(bin_width)
            bins = pd.date_range(
                data[data_col].min() - bin_width,
                data[data_col].max() + 2 * bin_width,
                freq=bin_width,
            )

        n_bins = len(bins) - 1
        data["bin"] = pd.cut(data[data_col], bins=bins, right=False)

    else:
        if not n_bins:
            if not bin_width:
                bin_width = 2 * iqr(data[data_col]) * len(data[data_col]) ** (-1 / 3)
            n_bins = int((data[data_col].max() - data[data_col].min()) / bin_width)

        bins = np.linspace(
            data[data_col].min() - bin_width,
            data[data_col].max() + 2 * bin_width,
            n_bins + 1,
        )

        data["bin"] = pd.cut(
            data[data_col],
            bins=bins,
            right=False,
        )

    binned_data = (
        (
            data.groupby([category_col, "bin"], observed=False)[weights_col]
            .sum()
            .reset_index(name="count")
        )
        if weights_col
        else (
            data.groupby([category_col, "bin"], observed=False)
            .size()
            .reset_index(name="count")
        )
    )

    binned_data[data_col] = (
        pd.to_datetime(binned_data["bin"].apply(lambda x: x.mid))
        if np.issubdtype(data[data_col].dtype, np.datetime64)
        else binned_data["bin"].apply(lambda x: x.mid).astype(float)
    )

    return binned_data


def create_kde_lines(
    data: pd.DataFrame,
    category_col: str,
    data_col: str,
    weights_col: str | None = None,
    n_points: int = 100,
    bandwidth: float | None = None,
) -> pd.DataFrame:
    data = data.dropna(subset=[data_col])

    kde_lines = []

    for category, group in data.groupby(category_col):
        x = group[data_col].to_numpy()
        weights = group[weights_col].to_numpy() if weights_col else None

        is_datetime = np.issubdtype(data[data_col].dtype, np.datetime64)
        if is_datetime:
            x = (pd.to_datetime(x) - pd.Timestamp("1970-01-01")) / pd.Timedelta(
                seconds=1
            )  # Convert datetime to numeric

        if len(x) < 2:
            x_min, x_max = x.min(), x.max() if len(x) > 0 else (0, 1)
            if is_datetime:
                x_min, x_max = (
                    pd.Timestamp(x_min, unit="s"),
                    pd.Timestamp(x_max, unit="s"),
                )
                x_grid = pd.date_range(start=x_min, end=x_max, periods=n_points)
                y_grid = np.zeros(len(x_grid))
            else:
                x_grid = np.linspace(x_min, x_max, n_points)
                y_grid = np.zeros_like(x_grid)

            kde_lines.append(
                pd.DataFrame(
                    {
                        category_col: category,
                        data_col: x_grid,
                        "density": y_grid,
                    }
                )
            )
            continue

        kde = gaussian_kde(x, weights=weights, bw_method=bandwidth)

        x_min, x_max = x.min(), x.max()
        x_grid = np.linspace(x_min, x_max, n_points)

        y_grid = kde(x_grid)

        if np.issubdtype(data[data_col].dtype, np.datetime64):
            x_grid = pd.to_datetime(x_grid, unit="s")

        kde_lines.append(
            pd.DataFrame(
                {
                    category_col: category,
                    data_col: x_grid,
                    "density": y_grid,
                }
            )
        )

    return pd.concat(kde_lines, ignore_index=True)


def create_outline(
    x: np.array, y: np.array, x_diff: np.array, smoothing
) -> tuple[np.array]:
    x_outline = np.repeat(x - x_diff / 2, 2)[1:]
    x_outline = np.concatenate([x_outline, [x.to_numpy()[-1] + x_diff[-1] / 2]])
    if smoothing:
        x_outline = x_outline + x_diff[-1] / 2
    y_outline = []
    for i in range(len(x)):
        y_outline.extend([y[i], y[i]])

    return x_outline, np.array(y_outline)


def color_to_rgb(color: str) -> tuple:
    if not isinstance(color, str):
        return color

    if color.startswith("#"):
        return hex_to_rgb(color)

    return name_to_rgb(color)


def _check_input(ridgetype, show_points):
    if ridgetype not in ["lines", "bins", None]:
        raise ValueError(f"Type should be 'lines' or 'bins'. recieved {ridgetype}")

    if show_points not in ["all", "outliers", "none", True, False]:
        raise ValueError(
            f"Incorrect show_points value: {show_points}. "
            "Should be one of 'all', 'outliers', 'none', True, False"
        )


def plot_points(
    fig,
    show_points,
    spread,
    ridgetype,
    initial_data,
    data_col,
    stats_col_init,
    initial_x,
    y,
    array_dict,
    categories_list,
    category,
    index,
    category_col,
    jitter_max_height,
    jitter_strength_x,
    jitter_strength_y,
    points_size,
    lower_fence,
    upper_fence,
    edgecolor,
    color,
    hoverdata,
    row,
    col,
):
    if show_points != "none" and show_points is not False:
        jitter_strength_x_new = jitter_strength_x * spread / 100
        jitter_amount_x = (
            pd.to_timedelta(
                np.random.uniform(
                    -jitter_strength_x_new, jitter_strength_x_new, len(initial_x)
                ),
                unit="s",
            )
            if np.issubdtype(initial_x.dtype, np.datetime64)
            else np.random.uniform(
                -jitter_strength_x_new, jitter_strength_x_new, len(initial_x)
            )
        )
        jitter_x = initial_x + jitter_amount_x

        if stats_col_init is None:
            jitter_amount_y = np.random.uniform(0, jitter_max_height, len(initial_x))
            jitter_y = (
                np.repeat(len(categories_list) - index - 1, len(initial_x))
                + jitter_amount_y
            )
        else:
            initial_y = (
                (
                    initial_data[initial_data[category_col] == category][stats_col_init]
                    * array_dict[f"{category}_norm_factor"]
                    * jitter_max_height
                    + len(categories_list)
                    - 1
                    - index
                )
                if ridgetype == "bins"
                else (
                    initial_data[initial_data[category_col] == category][stats_col_init]
                    * array_dict[f"{category}_norm_factor"]
                    * (
                        array_dict[f"y_{category}_normalized"].max()
                        / (
                            initial_data[initial_data[category_col] == category][
                                stats_col_init
                            ]
                            * array_dict[f"{category}_norm_factor"]
                        ).max()
                    )
                    * jitter_max_height
                    + len(categories_list)
                    - 1
                    - index
                )
                if array_dict[f"y_{category}_normalized"].max() != 0
                else (
                    initial_data[initial_data[category_col] == category][stats_col_init]
                    * 0
                    + jitter_max_height / 2
                    + len(categories_list)
                    - 1
                    - index
                )
            )
            jitter_strength_y_new = jitter_strength_y * y.max() / 100
            jitter_amount_y = np.random.uniform(
                -jitter_strength_y_new, jitter_strength_y_new, len(initial_y)
            )
            jitter_y = initial_y + jitter_amount_y

        if show_points == "outliers":
            outliers_mask = (initial_x < lower_fence) | (initial_x > upper_fence)
            outlier_x = initial_x[outliers_mask]

            if stats_col_init is not None:
                outlier_y = initial_y[outliers_mask]
            else:
                outlier_y = np.repeat(len(categories_list) - index - 1, len(outlier_x))

            jitter_x = outlier_x + jitter_amount_x[outliers_mask]
            jitter_y = outlier_y + jitter_amount_y[outliers_mask]

        if not stats_col_init:
            hovertemplate = "{{Data}}: %{x}".replace("{{Data}}", data_col)
            customdata = None
        elif not hoverdata:
            hovertemplate = "{{Data}}: %{x}<br>{{Stat}}: %{customdata}<br>".replace(
                "{{Data}}", data_col
            ).replace("{{Stat}}", stats_col_init)
            customdata = initial_data[initial_data[category_col] == category][
                stats_col_init
            ]
        else:
            hoverdata = [hoverdata] if isinstance(hoverdata, str) else hoverdata
            for col_ in hoverdata:
                if (
                    pd.api.types.is_categorical_dtype(initial_data[col_])
                    and "NA" not in initial_data[col_].cat.categories
                ):
                    initial_data[col_] = initial_data[col_].cat.add_categories("NA")
            customdata_rest = [
                initial_data[initial_data[category_col] == category][col].fillna("NA")
                for col in hoverdata
            ]
            customdata = np.stack(
                (
                    initial_data[initial_data[category_col] == category][
                        stats_col_init
                    ],
                    *customdata_rest,
                ),
                axis=-1,
            )
            hovertemplate = (
                "{{Data}}: %{x}<br>{{Stat}}: %{customdata[0]}<br>".replace(
                    "{{Data}}", data_col
                ).replace("{{Stat}}", stats_col_init)
            ) + "<br>".join(
                "{{col}}: %{customdata[{{c}}]}".replace("{{col}}", str(col)).replace(
                    "{{c}}", str(c + 1)
                )
                for c, col in enumerate(hoverdata)
            )

        fig.add_trace(
            go.Scatter(
                x=jitter_x,
                y=jitter_y,
                mode="markers",
                marker_size=points_size,
                marker_color=edgecolor if edgecolor else color,
                name=f"{category}",
                showlegend=False,
                hovertemplate=hovertemplate,
                customdata=customdata,
            ),
            row=row,
            col=col,
        )


def extend_x_y(x, y, supp_x_long, threshold_fraction=0.01):
    """
    Extends x and y by adding points to create smooth transitions to supp_x_long boundaries.

    Parameters:
        x (array-like): Original x values.
        y (array-like): Original y values.
        supp_x_long (array-like): Support x values defining extended boundaries.
        threshold_fraction (float): Fraction of x range to detect when the difference is too small.

    Returns:
        pd.Series, pd.Series: Extended x and y as Pandas Series.
    """
    # Convert inputs to Pandas Series
    x = pd.Series(x)
    y = pd.Series(y)
    supp_x_long = pd.Series(supp_x_long)

    # Check if x is of datetime dtype
    is_datetime = np.issubdtype(x.dtype, np.datetime64)

    if is_datetime:
        # Convert datetime to numeric (seconds since epoch)
        x_numeric = (x - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        supp_x_long_numeric = (supp_x_long - pd.Timestamp("1970-01-01")) / pd.Timedelta(
            seconds=1
        )
    else:
        x_numeric = x
        supp_x_long_numeric = supp_x_long

    # Find min and max of supp_x_long (numeric form)
    supp_min = supp_x_long_numeric.min()
    supp_max = supp_x_long_numeric.max()

    # Calculate the range of x for threshold
    x_range = x_numeric.max() - x_numeric.min()

    # Set the threshold based on the range of x (small fraction of range)
    threshold = threshold_fraction * x_range

    # Check if intervals are too small (avoid zero value artifacts)
    min_diff = x_numeric.min() - supp_min
    max_diff = supp_max - x_numeric.max()

    # Initialize empty arrays for pre and post extension
    pre_x, pre_y, post_x, post_y = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    # Generate 100 points for each required segment (avoid adding if the diff is too small)
    if (
        min_diff > threshold
    ):  # Only generate pre-extension if the difference is large enough
        pre_x = np.linspace(supp_min, x_numeric.min(), 100)
        pre_y = np.zeros_like(pre_x)

    if (
        max_diff > threshold
    ):  # Only generate post-extension if the difference is large enough
        post_x = np.linspace(x_numeric.max(), supp_max, 100)
        post_y = np.zeros_like(post_x)

    # Concatenate all parts (using numeric x)
    # Avoid adding redundant first or last extension points if already within range
    extended_x_numeric = []
    extended_y = []

    # Start with [supp_x_long.min(), 0] if it's different from x.min()
    if supp_min < x_numeric.min():
        extended_x_numeric.append(supp_min)
        extended_y.append(0)

    # Add pre-extension points
    extended_x_numeric.extend(pre_x)
    extended_y.extend(pre_y)

    # Add [x.min(), y[0]]
    extended_x_numeric.append(x_numeric.min())
    extended_y.append(y.iloc[0])

    # Original x and y
    extended_x_numeric.extend(x_numeric.values)
    extended_y.extend(y.values)

    # Add [x.max(), y[-1]]
    extended_x_numeric.append(x_numeric.max())
    extended_y.append(y.iloc[-1])

    # Add post-extension points
    extended_x_numeric.extend(post_x)
    extended_y.extend(post_y)

    # End with [supp_x_long.max(), 0] if it's different from x.max()
    if supp_max > x_numeric.max():
        extended_x_numeric.append(supp_max)
        extended_y.append(0)

    # If original x was datetime, convert extended_x back to datetime
    if is_datetime:
        extended_x = pd.to_datetime(extended_x_numeric, unit="s")
    else:
        extended_x = pd.Series(extended_x_numeric)

    extended_y = pd.Series(extended_y)

    return extended_x, extended_y


def create_line_hovertemplate(
    data_col,
    stats_col_init,
    initial_ridgetype,
    ridgetype,
    hover_stats,
    initial_x,
    upper_fence,
    lower_fence,
    y_outline,
    smoothing,
    array_dict,
    category,
    supp_x_long,
):
    hovertemplate = "{{Data}}: %{x}<br>{{Duration}}: %{customdata}".replace(
        "{{Data}}", data_col
    ).replace(
        "{{Duration}}",
        stats_col_init
        if stats_col_init and initial_ridgetype == "bins"
        else "count"
        if ridgetype == "bins"
        else "density",
    )

    hovertemplate = (
        hovertemplate
        if not hover_stats
        else (
            hovertemplate
            + "<br><br>"
            + "max: {{max}}<br>"
            "upper fence: {{upper_fence}}<br>"
            "q3: {{q3}}<br>"
            "median: {{median}}<br>"
            "mean: {{mean}}<br>"
            "q1: {{q1}}<br>"
            "lower fence: {{lower_fence}}<br>"
            "min: {{min}}".replace("{{max}}", str(initial_x.max()))
            .replace("{{upper_fence}}", str(upper_fence))
            .replace("{{q3}}", str(initial_x.quantile(0.75)))
            .replace("{{median}}", str(initial_x.median()))
            .replace("{{mean}}", str(initial_x.mean()))
            .replace("{{q1}}", str(initial_x.quantile(0.25)))
            .replace("{{lower_fence}}", str(lower_fence))
            .replace("{{min}}", str(initial_x.min()))
        )
    )
    customdata = (
        y_outline
        if initial_ridgetype == "bins" and smoothing == 0
        else extend_x_y(
            array_dict[f"x_{category}"], array_dict[f"y_{category}"], supp_x_long
        )[1]
    )

    return hovertemplate, customdata


def _default_ridgetype(ridgetype, stats_col, smoothing):
    if not ridgetype:
        if stats_col:
            ridgetype = "bins"
            smoothing = 0.9
        else:
            ridgetype = "lines"
    return ridgetype, smoothing


def ridgeline(
    data: pd.DataFrame,
    *,
    category_col: str,
    categories_order: list | None,
    data_col: str = "Data",
    data_range: tuple | None = None,
    stats_col: str | None = None,
    ridgetype: str | None = None,
    smoothing: float | None = None,
    normalize_each: bool = True,
    scaling_factor: float = 1.75,
    bin_width: float | str | None = None,
    n_bins: int | None = None,
    bandwidth: float | None = None,
    n_points: int = 100,
    edgecolor: str | None = None,
    colorway: list | None = None,
    opacity: float = 0.5,
    hover_stats: bool = True,
    hoverdata: str | list | None = None,
    line_width: int = 2,
    show_points: str | bool = "all",
    points_size: float = 5,
    jitter_max_height: float = 0.5,
    jitter_strength_y: float = 0,
    jitter_strength_x: float = 0,
    ylabels_xref: str | None = None,
    ylabels_name: str = "ytitles",
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    subplot_n: int | None = None,
) -> go.Figure:
    """
    Generates a ridgeline plot in multiple modes: KDE ridgelines, histogram-like ridgelines,
    or combined scatter-ridgeline plots. The mode depends on the presence of `stats_col`
    and the `ridgetype` parameter.

    **Modes**:
    - `ridgetype="lines"`: KDE ridgeline (weighted if `stats_col`) with optional jittered scatter
        points / scatter plot normalized by y_max / kde_max if `stats_col`.
    - `ridgetype="bins"`: ridgeline histogram (weighted if `stats_col`) smoothed to a line with
        points at the center of bins if `smoothing` is provided with optional jittered scatter
        points / scatter plot using `stats_col` for y-coordinates if `stats_col`.

    **Default Behavior**:
    - If `stats_col` is not specified: Generates a KDE ridgeline (`ridgetype="lines"`).
    - If `stats_col` is specified: Generates a smoothed histogram-like ridgeline (`ridgetype="bins"`)
      with `smoothing=0.9` by default for smoother curves.

    Parameters:
    ----------
    data : pd.DataFrame
        Input data for plotting.

    category_col : str
        Column containing categories for the ridgelines. Categories will be used as y-labels.

    categories_order : list
        List of categories in desired order.

    data_col : str, default="Data"
        Column containing x-values (can be numeric or datetime).

    data_range : tuple or None
        The range of x-values to include in the plot.
        If `None`, the range is inferred from the data.

    stats_col : str or None, default=None
        Column containing y-values. When specified, controls scatter y-values and ridgeline weighting.

    ridgetype : {"lines", "bins"} or None, default="lines" if `stats_col` is None, otherwise "bins"
        - "lines": KDE ridgelines.
        - "bins": Histogram-like ridgelines (supports smoothing for a curved look).

    smoothing : float or None, default=None
        - For `ridgetype="lines"`, defaults to 0 (no effect).
        - For `ridgetype="bins"`, determines the smoothness of the curve. If None, produces histogram bars.

    normalize_each : bool, default=True
        - If True, normalizes ridgelines separately so their areas are equal.
        - If False, uses global min-max normalization across all data.

    scaling_factor : float, default=1.75
        Specifies the maximum height of ridgelines (1 corresponds to a single category height).

    bin_width : float, str, or None, default=None
        Specifies bin width for `ridgetype="bins"`. For datetime, can be strings like "1D".

    n_bins : int or None, default=None
        Number of bins for `ridgetype="bins"`.

    bandwidth : float or None, default=None
        KDE bandwidth for `ridgetype="lines"`.

    n_points : int, default=100
        Number of points in the KDE grid for `ridgetype="lines"`.

    edgecolor : str or None, default=None
        Color of ridgeline edges and points. If None, uses the same color as the fill area.

    colorway : list or None, default=None
        List of colors for the ridgeline areas and lines.

    opacity : float, default=0.5
        Opacity of the area under ridgelines.

    hover_stats : bool, default=True
        If True, displays statistical summary (e.g., median, mean) on ridgeline hover.

    hoverdata : str or list or None, default=None
        Additional data columns to display on hover for scatter points.

    show_points : {"all", "none", "outliers"} or bool, default="all"
        Controls the visibility of individual scatter points.

    points_size : float, default=5
        Size of scatter points.

    jitter_max_height : float, default=0.5
        - When `stats_col` is not specified, it determines the maximum y-range for jitter,
            with 1 representing a range from the category baseline the category height.
        - When `stats_col` is specified, it represents the ratio of the scatter y-scale to the ridgeline y-scale.

    jitter_strength_y : float or None, default=0
        Jitter strength in the y-direction as a percentage of the category's y-range
        (used only when `stats_col` is specified).

    jitter_strength_x : float or None, default=0
        Jitter strength in the x-direction as a percentage of the data's x-range.

    ylabels_xref : str or None, default=None
        x-reference for y-label annotations. Useful for subplot placement.

    ylabels_name : str, default="ytitles"
        Name for y-label annotations. Can be updated using `fig.update_annotations`.

    fig : go.Figure or None, default=None
        Existing figure to add the ridgeline plot to. If None, creates a new figure.

    row : int or None, default=None
        Row index for subplot placement.

    col : int or None, default=None
        Column index for subplot placement.

    subplot_n : int or None, default=None
        Index for subplot annotations when using `xref` and `yref`.

    Returns:
    -------
    go.Figure
        A Plotly figure object containing the ridgeline plot.
    """

    # add points_color, points_edgewidth, points_edgecolor

    _check_input(ridgetype, show_points)

    ridgetype, smoothing = _default_ridgetype(ridgetype, stats_col, smoothing)

    initial_data = data
    initial_ridgetype = ridgetype
    stats_col_init = stats_col

    if ridgetype == "bins":
        data = create_histogram(
            data, category_col, data_col, stats_col, bin_width, n_bins
        )
        stats_col = "count"
    elif ridgetype == "lines":
        data = create_kde_lines(
            data, category_col, data_col, stats_col, n_points, bandwidth
        )
        smoothing = 0
        stats_col = "density"

    if ridgetype == "bins" and smoothing is not None:
        ridgetype = "lines"
    else:
        smoothing = 0

    data = data.rename(columns={stats_col: "count", data_col: "Data"})
    array_dict = _create_array_dict(
        data, category_col, "Data", normalize_each, scaling_factor
    )
    categories_list = (
        categories_order
        if categories_order is not None
        else sorted(data[category_col].unique())
    )

    fig = fig if fig else go.Figure()
    for index, category in enumerate(categories_list):
        x = array_dict[f"x_{category}"]
        y = array_dict[f"y_{category}_normalized"].to_numpy()
        color = (
            colorway[index]
            if colorway
            else pio.templates[pio.templates.default].layout.colorway[index]
        )

        x_diff = np.diff(x)
        x_diff = np.concatenate(([x_diff[0]], x_diff))

        supp_x_small = (
            [
                array_dict[f"x_{category}"].min() - x_diff[-1] / 2,
                array_dict[f"x_{category}"].max() + x_diff[-1] / 2,
            ]
            if (ridgetype == "bins") and not smoothing
            else [array_dict[f"x_{category}"].min(), array_dict[f"x_{category}"].max()]
        )

        supp_x_long = (
            (
                [
                    data["Data"].min() - x_diff[-1] / 2,
                    data["Data"].max() + x_diff[-1] / 2,
                ]
                if (ridgetype == "bins") and not smoothing
                else [data["Data"].min(), data["Data"].max()]
            )
            if not data_range
            else (
                [
                    min(data["Data"].min() - x_diff[-1] / 2, data_range[0]),
                    max(data["Data"].max() + x_diff[-1] / 2, data_range[1]),
                ]
                if (ridgetype == "bins") and not smoothing
                else [*data_range]
            )
        )

        if initial_ridgetype == "bins":
            _x, y_outline = create_outline(
                x, array_dict[f"y_{category}"].to_numpy(), x_diff, smoothing
            )
            _x, y_outline = extend_x_y(_x, y_outline, supp_x_long)
        x, y = (
            (x, y)
            if (ridgetype == "lines") or smoothing
            else create_outline(x, y, x_diff, smoothing)
        )
        x, y = extend_x_y(x, y, supp_x_long)

        initial_x = initial_data[initial_data[category_col] == category][data_col]
        if bin_width == "1D":
            initial_x = (
                initial_x + x_diff[-1] / 2 if initial_ridgetype == "bins" else initial_x
            )
        upper_fence = initial_x.quantile(0.75) + 1.5 * (
            initial_x.quantile(0.75) - initial_x.quantile(0.25)
        )
        lower_fence = initial_x.quantile(0.25) - 1.5 * (
            initial_x.quantile(0.75) - initial_x.quantile(0.25)
        )

        hovertemplate, customdata = create_line_hovertemplate(
            data_col,
            stats_col_init,
            initial_ridgetype,
            ridgetype,
            hover_stats,
            initial_x,
            upper_fence,
            lower_fence,
            y_outline,
            smoothing,
            array_dict,
            category,
            supp_x_long,
        )

        # small supporting lines for drawing areas
        fig.add_trace(
            go.Scatter(
                x=supp_x_small,
                y=np.full(2, len(categories_list) - 1 - index),
                mode="lines",
                line_width=1,
                hoverinfo="skip",
                showlegend=False,
                line_color="#eaeaea",
            ),
            row=row,
            col=col,
        )

        # drawing lines and areas
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y + len(categories_list) - index - 1,
                mode="lines",
                line_shape="spline",
                line_smoothing=smoothing,
                fillcolor=f"rgba{(*color_to_rgb(color), opacity)}",
                fill="tonexty",
                line=dict(color=edgecolor if edgecolor else color, width=line_width),
                name=f"{category}",
                showlegend=False,
                hovertemplate=hovertemplate,
                customdata=customdata,
            ),
            row=row,
            col=col,
        )

        spread = (
            (
                initial_data[data_col].max() - initial_data[data_col].min()
            ).total_seconds()
            if np.issubdtype(initial_x.dtype, np.datetime64)
            else (initial_data[data_col].max() - initial_data[data_col].min())
        )

        plot_points(
            fig,
            show_points,
            spread,
            initial_ridgetype,
            initial_data,
            data_col,
            stats_col_init,
            initial_x,
            y,
            array_dict,
            categories_list,
            category,
            index,
            category_col,
            jitter_max_height,
            jitter_strength_x,
            jitter_strength_y,
            points_size,
            lower_fence,
            upper_fence,
            edgecolor,
            color,
            hoverdata,
            row=row,
            col=col,
        )

        # y-labels
        if not ylabels_xref and (subplot_n or row or col):
            fig.add_annotation(
                x=data_range[0] if data_range else data["Data"].min(),
                y=len(categories_list) - index - 1,
                xanchor="right",
                text=f"{category}    ",
                showarrow=False,
                yshift=4,
                xref=f"x{subplot_n}" if subplot_n else "x",
                yref=f"y{subplot_n}" if subplot_n else "y",
                name="ytitles",
            )
        else:
            fig.add_annotation(
                x=0,
                y=len(categories_list) - index - 1,
                xanchor="right",
                text=f"{category}",
                showarrow=False,
                yshift=4,
                xref=ylabels_xref if ylabels_xref else "paper",
                yref=f"y{subplot_n}" if subplot_n else "y",
                name=ylabels_name,
            )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=data_col, showline=False)
    fig.update_yaxes(showticklabels=False, showline=False, zeroline=False)
    return fig
