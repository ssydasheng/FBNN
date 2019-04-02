_DATA = dict()

def register(name):

  def add_to_dict(fn):
    global _DATA
    _DATA[name] = fn
    return fn

  return add_to_dict


def get_data(name):
  """Fetches a merged group of hyperparameter sets (chronological priority)."""
  return _DATA[name]
