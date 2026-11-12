package dev.argo.imagelab.core

/**
 * Registry function ids the app's buttons call through [LabApi.call] (docs/INTERFACES.md §2):
 * a button and an agent step are the same code path. Kept in one place so a rename in the
 * registry is a one-line change here.
 */
object Fn {
    const val LIBRARY_QUERY = "library.query"
    const val LIBRARY_IMPORT = "library.import"
    const val LIBRARY_SET_RATING = "library.setRating"
    const val LIBRARY_SET_FLAG = "library.setFlag"
    const val BATCH_EXPORT = "batch.export"

    const val DOC_INSPECT = "doc.inspect"
    const val DOC_OPEN = "doc.open"
    const val DOC_HISTORY = "doc.history"
    const val DOC_HISTORY_GO = "doc.historyGo"
    const val DOC_SNAPSHOT = "doc.snapshot"
    const val DOC_RESTORE_SNAPSHOT = "doc.restoreSnapshot"
    const val DOC_DELETE_SNAPSHOT = "doc.deleteSnapshot"
    const val DOC_ORIGINAL = "doc.original"
    const val DOC_EXPORT = "doc.export"

    /** `{id, params}`: one allowlisted PhotoCraft engine command. Masked edits load the mask as the selection first ([SELECT_FROM_MASK]). */
    const val EDIT_RUN = "edit.run"
    const val EDIT_ADJUST = "edit.adjust"
    const val EDIT_LOOK = "edit.look"
    const val EDIT_LOOK_SUGGEST = "edit.lookSuggest"
    const val EDIT_FACE_LIQUIFY = "edit.faceLiquify"
    const val EDIT_PREVIEW = "edit.preview"
    const val EDIT_STRAIGHTEN = "edit.straighten"

    const val SELECT_SUBJECT = "select.subject"
    const val SELECT_SKY = "select.sky"
    const val SEGMENT_POINT = "segment.point"
    const val SEGMENT_BOX = "segment.box"
    const val SEGMENT_CONCEPT = "segment.text"
    const val SELECT_FROM_MASK = "select.fromMask"
    const val CV_READ = "cv.read"
    const val CV_TO_LAYER = "cv.toDocumentLayer"

    const val CV_INVERT_MASK = "cv.invertMask"
    const val CV_DENOISE = "cv.denoise"
}
