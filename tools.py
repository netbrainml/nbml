def shapes(*x):
    for idx, item in enumerate(x): print(f"arg_{idx}: {item.shape}")