package dev.argo.imagelab

import dev.argo.imagelab.core.Imports
import dev.argo.imagelab.core.LabEvent
import dev.argo.imagelab.ui.WidthClass
import dev.argo.imagelab.ui.widthClassOf
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AppLogicTest {
    @Test
    fun parsesCoreEvents() {
        assertEquals(LabEvent.DocChanged(0, 7, "agent"), LabEvent.parse("""{"type":"docChanged","doc":0,"revision":7,"source":"agent"}"""))
        assertEquals(LabEvent.LibraryChanged(listOf("a", "b"), "mcp"), LabEvent.parse("""{"type":"libraryChanged","photos":["a","b"],"source":"mcp"}"""))
        assertEquals(LabEvent.JobProgress("j1", 3, 10, "exporting"), LabEvent.parse("""{"type":"jobProgress","job":"j1","done":3,"total":10,"message":"exporting"}"""))
        assertEquals(LabEvent.JobFinished("j1", true, "10 exported"), LabEvent.parse("""{"type":"jobFinished","job":"j1","ok":true,"summary":"10 exported"}"""))
        assertNull(LabEvent.parse("""{"type":"somethingNew"}"""))
        assertNull(LabEvent.parse("not json"))
    }

    @Test
    fun sanitizesPickedNames() {
        assertEquals("passwd", Imports.sanitize("../../etc/passwd"))
        assertEquals("my photo_1_.jpg", Imports.sanitize("my photo(1).jpg"))
        assertEquals("photo.jpg", Imports.sanitize(""))
        assertEquals("hidden.jpg", Imports.sanitize(".hidden.jpg"))
    }

    @Test
    fun widthClassesFollowMaterialBreakpoints() {
        assertEquals(WidthClass.Compact, widthClassOf(411f, 914f))
        assertEquals(WidthClass.Medium, widthClassOf(720f, 960f))
        assertEquals(WidthClass.Expanded, widthClassOf(1104f, 920f))
    }
}
