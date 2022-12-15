import torch
from utils.networks import Mnih2015
from pathlib import Path

def convert_model(in_model: Path, out_model: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(in_model, map_location=device)
    model.eval()

    torch.save(model.state_dict(), out_model)

def bulk_model_conversion(in_dir: str, out_dir: str) -> None:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    for model in in_dir.glob('*.pt'):
        convert_model(model, out_dir.joinpath(model.name+'.pth'))