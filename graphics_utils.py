import scipy.stats as stats


def z_score_from_percentage(percentage):
    """Return the standard-normal cutoff with upper-tail probability percentage / 100."""
    proportion = percentage / 100

    z_score = stats.norm.ppf(1 - proportion)

    return z_score
