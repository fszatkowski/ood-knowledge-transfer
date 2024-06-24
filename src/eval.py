import argparse
from pathlib import Path

import torch

from dotenv import load_dotenv
from pytorch_lightning import Trainer
import pandas as pd

from distiller import ImgDistill
from data_utils import prepare_data

parser = argparse.ArgumentParser(description="Knowledge Distillation evaluation.")

parser.add_argument("--gpus", default=-1, type=int, help="how many gpus to use")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size to use")
parser.add_argument("--workers", default=8, type=int, help="number of workers")

# Teacher settings
parser.add_argument("--teacher_arch", default="resnet18", type=str, help="arch for teacher")
parser.add_argument("--teacher_ckpt", default="", type=str, help="ckpt to load teacher. not needed for IN-1k")

# Student
parser.add_argument("--student_arch", default="resnet50", type=str, help="arch for student")
parser.add_argument("--student_dir", default="", type=str, help="Directory to load student models from.")

parser.add_argument("--dst_dataset", default="cifar100", type=str, help="Dataset to use for distillation")
parser.add_argument("--tgt_dataset", default="cifar100", type=str, help="Dataset to use for evaluation")

parser.add_argument("--output_file", default='eval_output.csv', type=str, help="Output eval filename")


if __name__ == "__main__":
    args = parser.parse_args()

    load_dotenv()

    dst_loader, tgt_loader, num_classes = prepare_data(args)

    trainer = Trainer(
        accelerator="cuda",
        devices=args.gpus,
        max_epochs=0,
        callbacks=[],
    )
    # training module with teacher and student and optimizer
    distiller = ImgDistill(
        num_classes=num_classes,
        teacher_arch=args.teacher_arch,
        student_arch=args.student_arch,
        teacher_ckpt=args.teacher_ckpt,
    )

    # TODO allow evaluation with mismatched architectures
    ckpt_src = torch.load(args.teacher_ckpt, map_location='cpu')['state_dict']
    ckpt_src = {k.replace('model.', ''): v for k, v in ckpt_src.items() if 'model' in k}
    distiller.student.load_state_dict(ckpt_src)

    results = []
    results_src = trainer.test(distiller, dataloaders=tgt_loader)[0]
    results_src['model'] = 'teacher'
    results.append(results_src)

    student_ckpts = Path(args.student_dir).rglob('last.ckpt')
    for ckpt in student_ckpts:
        ckpt_tgt = torch.load(ckpt, map_location='cpu')['state_dict']
        ckpt_tgt = {k.replace('student.', ''): v for k, v in ckpt_tgt.items() if 'student' in k}
        distiller.student.load_state_dict(ckpt_tgt)

        results_student = trainer.test(distiller, dataloaders=tgt_loader)[0]
        results_student['model'] = str(ckpt)
        results.append(results_student)

    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'test/acc_top1', 'test/acc_top5', 'test/loss']]
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)