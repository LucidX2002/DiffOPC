# save edge params

# plt.show()

# fig, axs = plt.subplots(2, 4, figsize=(20, 12))
# if len(all_mask_edges) >= 8:
#     all_mask_edges = all_mask_edges[-8:]
# for i, ax in enumerate(axs.flat):
#     if i < len(all_mask_edges):
#         ax.imshow(all_mask_edges[i]["mask"])
#         ax.set_title(f"Iteration {all_masks[i]['iteration']}")
# plt.tight_layout()
# plt.savefig(f"{str(save_dir)}/EdgeILT_M1_test{case_id}_edge.png", dpi=300)

# for m in all_mask_edges:
#     plt.imsave(
#         f"{str(save_dir)}/EdgeILT_test{idx}_edge_{m['iteration']}.png",
#         m["mask"],
#         dpi=300,
#     )
# plt.show()
