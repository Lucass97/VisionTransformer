import torch
import matplotlib.pyplot as plt


def apply_color_palette(attn_map, palette='viridis'):
    """
    Applica una color palette al tensore attn_map.
    :param attn_map: Tensor di attenzione (batch, channels, height, width)
    :param palette: Nome della palette di colori (ad esempio 'viridis', 'inferno', 'plasma', ecc.)
    :return: Immagine colorata in formato (batch, channels, height, width)
    """
    batch_size, n_channels, height, width = attn_map.shape

    # Convertiamo attn_map in un array numpy per applicare la mappa di colori
    attn_map_np = attn_map.squeeze().cpu().numpy()

    # Creiamo una mappa di colori usando matplotlib
    colormap = plt.get_cmap(palette)
    
    # Normalizziamo i valori in attn_map tra 0 e 1 per la mappa di colori
    attn_map_norm = (attn_map_np - attn_map_np.min()) / (attn_map_np.max() - attn_map_np.min())
    
    # Applichiamo la palette per ogni elemento nel batch
    colored_maps = []
    for i in range(batch_size):
        # Applichiamo il colormap e otteniamo un'immagine RGB
        colored_map = colormap(attn_map_norm[i])  # (height, width, 4) -> RGBA
        colored_map = colored_map[:, :, :3]  # Solo RGB (ignora alpha)

        # Convertiamo in tensor PyTorch e modifichiamo la forma per ottenere (C, H, W)
        colored_map = torch.tensor(colored_map).permute(2, 0, 1)  # Cambiamo forma in (C, H, W)
        colored_maps.append(colored_map.unsqueeze(0))  # Aggiungiamo il batch dimension
    
    # Concatenazione delle immagini colorate
    colored_maps = torch.cat(colored_maps, dim=0)

    return colored_maps